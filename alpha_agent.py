"""
AlphaAgent: LLM policy that maps observations → actions.

Design principles (vs. the naive refine.py):
  - System prompt is *dynamic*: built from failing checks + current trajectory
  - Uses structured JSON output with chain-of-thought reasoning
  - Maintains a *population* of candidates (not just one thread)
  - Has three modes: EXPLORE (high temperature), EXPLOIT (low temp), DIVERSIFY (cross-breed)
  - Operator library is injected lazily based on what's relevant to the current state
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from textwrap import dedent
from typing import Optional

from google import genai
from google.genai import types


# ─────────────────────────────────────────────────────────────────────────────
#  Operator & system instruction loader
# ─────────────────────────────────────────────────────────────────────────────

OPERATORS_DIR = Path(__file__).parent / "operators"
SYSTEM_INSTRUCTIONS_DIR = Path(__file__).parent / "system_instructions"


def load_operator_docs(categories: list[str]) -> str:
    """Load only the operator categories relevant to the current task."""
    parts = []
    for cat in categories:
        path = OPERATORS_DIR / f"{cat}.txt"
        if path.exists():
            parts.append(f"## {cat} Operators\n{path.read_text().strip()}")
    return "\n\n".join(parts)


def load_warnings() -> str:
    path = SYSTEM_INSTRUCTIONS_DIR / "warnings.txt"
    if path.exists():
        return path.read_text().strip()
    return ""


# ─────────────────────────────────────────────────────────────────────────────
#  Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

class PromptBuilder:
    """
    Builds highly targeted prompts based on the current episode state.
    Unlike refine.py's static system prompt, this adapts every call.
    """

    # Map failing check names → recommended operator categories to inject
    CHECK_TO_OPERATORS = {
        "HIGH_TURNOVER":          ["TimeSeries", "Transformational"],
        "LOW_TURNOVER":           ["TimeSeries"],
        "LOW_FITNESS":            ["CrossSectional", "TimeSeries"],
        "CONCENTRATED_WEIGHT":    ["CrossSectional", "Group"],
        "LOW_SUB_UNIVERSE_SHARPE":["Group", "CrossSectional"],
        "LOW_SHARPE":             ["Arithmetic", "TimeSeries", "CrossSectional"],
    }

    BASE_OPERATOR_CATEGORIES = ["Arithmetic", "CrossSectional", "TimeSeries"]

    @classmethod
    def build_system_prompt(
        cls,
        datafield_context: str,
        failing_checks: list[str],
        mode: str,
        operator_override: Optional[list[str]] = None,
    ) -> str:
        # Determine which operator docs to inject
        cats = set(cls.BASE_OPERATOR_CATEGORIES)
        for check in failing_checks:
            cats.update(cls.CHECK_TO_OPERATORS.get(check, []))
        if operator_override:
            cats.update(operator_override)

        operator_docs = load_operator_docs(sorted(cats))
        warnings = load_warnings()

        mode_guidance = {
            "explore": dedent("""
                ## MODE: EXPLORE
                You are in exploration mode. Be creative and try novel signal architectures.
                Do not be afraid to rewrite the expression completely.
                Think about underexplored combinations of operators and data fields.
                Temperature is high — embrace this freedom productively.
            """),
            "exploit": dedent("""
                ## MODE: EXPLOIT
                You are in exploitation mode. The signal is promising — improve it incrementally.
                Make the SMALLEST change that fixes the failing check.
                Do NOT rewrite expressions that are already working.
                Focus on wrapping or adjusting, not replacing.
            """),
            "diversify": dedent("""
                ## MODE: DIVERSIFY
                You are cross-breeding two high-scoring parent alphas.
                Your job is to intelligently combine their signal logic — not just concatenate them.
                Think about what each parent captures and how to get complementary exposure.
            """),
        }[mode]

        return dedent(f"""
            You are a senior quantitative researcher at a top systematic hedge fund.
            Your job is to generate and refine alpha expressions for WorldQuant BRAIN.

            An alpha is a mathematical formula that predicts future stock returns.
            It is evaluated by simulating it against historical data.

            {mode_guidance}

            ---

            {operator_docs}

            ---

            {datafield_context}

            ---

            {warnings}

            ---

            ## Output Contract
            You MUST return valid JSON with ALL of these fields:
            - "chain_of_thought": Your step-by-step reasoning (be specific about WHY)
            - "action_type": One of: "mutate_expression", "change_settings", "radical_rewrite", "cross_breed"
            - "expression": The FASTEXPR alpha string (syntactically correct, semicolons between statements)
            - "decay": Integer 0-20
            - "neutralization": One of: NONE, MARKET, SECTOR, INDUSTRY, SUBINDUSTRY
            - "truncation": Float 0.01-0.10
            - "universe": One of: TOP3000, TOP1000, TOP500, TOP200
            - "confidence": Float 0.0-1.0 (your confidence this will improve the alpha)
        """).strip()

    @classmethod
    def build_user_message(
        cls,
        step: int,
        current_expression: str,
        current_settings: dict,
        sim_summary: str,
        history_summary: str,
        parent_alphas: Optional[list[dict]] = None,
    ) -> str:
        parent_section = ""
        if parent_alphas:
            parent_section = "\n\n## Parent Alphas (for cross-breeding)\n"
            for i, p in enumerate(parent_alphas):
                parent_section += f"\nParent {i+1} (Sharpe={p['sharpe']:.3f}):\n{p['expression']}\n"

        return dedent(f"""
            ## Step {step}

            ### Current Alpha Expression
            ```
            {current_expression}
            ```

            ### Current Settings
            {json.dumps(current_settings, indent=2)}

            ### Simulation Results
            {sim_summary}

            ### Trajectory Summary
            {history_summary}
            {parent_section}

            Based on the above, generate your next action.
            Remember: return ONLY valid JSON matching the Output Contract.
        """).strip()


# ─────────────────────────────────────────────────────────────────────────────
#  Agent
# ─────────────────────────────────────────────────────────────────────────────

class AlphaAgent:
    """
    LLM-powered agent that generates alpha actions.

    Features vs. baseline refine.py:
      - Dynamic system prompts (targeted to failing checks)
      - Multi-key rotation with exponential backoff
      - Three generation modes (explore / exploit / diversify)
      - Structured JSON output with chain-of-thought
      - Confidence-weighted action selection
    """

    RESPONSE_SCHEMA = {
        "type": "object",
        "required": [
            "chain_of_thought", "action_type", "expression",
            "decay", "neutralization", "truncation", "universe", "confidence"
        ],
        "properties": {
            "chain_of_thought":  {"type": "string"},
            "action_type":       {"type": "string"},
            "expression":        {"type": "string"},
            "decay":             {"type": "integer"},
            "neutralization":    {"type": "string"},
            "truncation":        {"type": "number"},
            "universe":          {"type": "string"},
            "confidence":        {"type": "number"},
        }
    }

    def __init__(
        self,
        api_keys: list[str],
        model: str = "gemini-2.5-flash",
        explore_temp: float = 0.9,
        exploit_temp: float = 0.2,
    ):
        self.api_keys = api_keys
        self.key_idx = 0
        self.model = model
        self.explore_temp = explore_temp
        self.exploit_temp = exploit_temp
        self._client = self._make_client()

    def _make_client(self) -> genai.Client:
        return genai.Client(api_key=self.api_keys[self.key_idx])

    def _rotate_key(self):
        self.key_idx = (self.key_idx + 1) % len(self.api_keys)
        self._client = self._make_client()

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        mode: str = "exploit",
        max_retries: int = 5,
    ) -> dict:
        """Call Gemini with retry/key-rotation logic. Returns parsed dict."""
        temperature = self.explore_temp if mode == "explore" else self.exploit_temp

        config = types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=types.Schema(
                type=types.Type.OBJECT,
                required=self.RESPONSE_SCHEMA["required"],
                properties={
                    k: types.Schema(type=types.Type.STRING)
                    if v["type"] == "string"
                    else types.Schema(type=types.Type.NUMBER)
                    if v["type"] == "number"
                    else types.Schema(type=types.Type.INTEGER)
                    for k, v in self.RESPONSE_SCHEMA["properties"].items()
                },
            ),
            system_instruction=[types.Part.from_text(text=system_prompt)],
        )

        for attempt in range(max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=[types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=user_message)]
                    )],
                    config=config,
                )
                if response.parsed:
                    return dict(response.parsed)
                # Try manual JSON parse from text
                text = response.text or ""
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except Exception as e:
                wait = 2 ** attempt
                print(f"[Agent] Attempt {attempt+1} failed: {e}. Rotating key, waiting {wait}s...")
                self._rotate_key()
                time.sleep(wait)

        raise RuntimeError("AlphaAgent: all retries exhausted")