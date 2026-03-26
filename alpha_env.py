"""
AlphaEnv: Core data types, reward function, and hard setting constraints.

Hard locks enforced here (cannot be overridden by LLM):
  - universe always TOP3000
  - decay always >= 1
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
#  Hard constraints
# ─────────────────────────────────────────────────────────────────────────────

HARD_LOCKED_UNIVERSE = "TOP3000"
DECAY_MINIMUM        = 1


def enforce_hard_constraints(settings: dict) -> dict:
    """Apply hard locks before any simulation. LLM cannot override these."""
    corrected = dict(settings)

    if corrected.get("universe") != HARD_LOCKED_UNIVERSE:
        print(f"[HardLock] universe={corrected.get('universe')} -> {HARD_LOCKED_UNIVERSE}")
        corrected["universe"] = HARD_LOCKED_UNIVERSE

    try:
        if int(corrected.get("decay", DECAY_MINIMUM)) < DECAY_MINIMUM:
            print(f"[HardLock] decay={corrected.get('decay')} -> {DECAY_MINIMUM}")
            corrected["decay"] = DECAY_MINIMUM
    except (TypeError, ValueError):
        corrected["decay"] = DECAY_MINIMUM

    return corrected


# ─────────────────────────────────────────────────────────────────────────────
#  Domain types
# ─────────────────────────────────────────────────────────────────────────────

class ActionType(str, Enum):
    MUTATE_EXPRESSION  = "mutate_expression"
    CHANGE_SETTINGS    = "change_settings"
    RADICAL_REWRITE    = "radical_rewrite"
    CROSS_BREED        = "cross_breed"
    WAIT               = "wait"


@dataclass
class AlphaAction:
    action_type: ActionType
    expression:  Optional[str]  = None
    settings:    Optional[dict] = None
    reasoning:   Optional[str]  = None
    parent_ids:  Optional[list] = None
    confidence:  float          = 0.5


@dataclass
class SimResult:
    alpha_id:   str
    sharpe:     float
    fitness:    float
    turnover:   float
    returns:    float
    drawdown:   float
    checks:     list[dict]
    expression: str  = ""     # stored for expression-aware fingerprinting
    raw:        dict = field(default_factory=dict)

    @property
    def passing(self) -> bool:
        skip = {"MATCHES_PYRAMID", "MATCHES_THEMES"}
        return all(
            c["result"] in ("PASS", "PENDING")
            for c in self.checks
            if c["name"] not in skip
        )

    @property
    def failing_checks(self) -> list[str]:
        skip = {"MATCHES_PYRAMID", "MATCHES_THEMES"}
        return [
            c["name"] for c in self.checks
            if c["name"] not in skip
            and c["result"] not in ("PASS", "PENDING")
        ]

    @property
    def fingerprint(self) -> str:
        """
        Expression-aware fingerprint (suggestion #4 from Gemini Pro review).
        Stats hash alone can collide — two different expressions producing
        identical metrics would be wrongly deduplicated. Expression hash prevents this.
        """
        stats_str = f"{self.sharpe:.3f}|{self.fitness:.3f}|{self.turnover:.3f}"
        expr_hash = hashlib.md5(self.expression.strip().encode()).hexdigest()[:8]
        return f"{stats_str}|{expr_hash}"


@dataclass
class AlphaObservation:
    episode_id:      str
    step:            int
    sim_result:      Optional[SimResult]
    reward:          float
    done:            bool
    info:            dict
    system_message:  str
    history_summary: str


@dataclass
class AlphaState:
    episode_id:      str
    difficulty:      str
    best_sharpe:     float = 0.0
    best_alpha_id:   str   = ""
    step_count:      int   = 0
    sim_budget:      int   = 20
    plateau_count:   int   = 0
    visited_fps:     set   = field(default_factory=set)
    history:         list  = field(default_factory=list)
    target_achieved: bool  = False


# ─────────────────────────────────────────────────────────────────────────────
#  Hard constraint message builder (suggestion #1 from Gemini Pro review)
# ─────────────────────────────────────────────────────────────────────────────

def build_hard_constraint(failing_checks: list[str], result: Optional[SimResult]) -> str:
    """
    Injects specific imperative instructions at the TOP of the user message
    based on the exact failing check. Much stronger than burying rules in system prompt.
    """
    if not failing_checks or result is None:
        return ""

    constraints = []

    if "LOW_FITNESS" in failing_checks:
            has_wrapper = any(kw in result.expression for kw in ["quantile(", "zscore(rank("])
            if not has_wrapper:
                constraints.append(
                    "[LOW_FITNESS] No gaussian wrapper found. "
                    "MUST add quantile(signal, driver=\"gaussian\") as final line. "
                    "Change nothing else."
                )
            elif result.turnover > 0.40:
                # FIX: Force mathematical smoothing instead of infinitely increasing settings decay
                constraints.append(
                    f"[LOW_FITNESS] Wrapper present, turnover={result.turnover:.3f} > 0.40. "
                    f"MUST use action_type=mutate_expression. "
                    f"Inject ts_decay_linear(..., 5) or hump() BEFORE the quantile() wrapper "
                    f"to reduce turnover mathematically. Do NOT just increase the decay setting."
                )
            elif result.turnover > 0.30:
                constraints.append(
                    f"[LOW_FITNESS] Wrapper present, turnover={result.turnover:.3f}. "
                    f"Try swapping quantile() to zscore(rank()) or add a hump() filter."
                )
            else:
                constraints.append(
                    "[LOW_FITNESS] Turnover ok. Try swapping quantile() to zscore(rank()) or vice versa."
                )

    if "HIGH_TURNOVER" in failing_checks:
        constraints.append(
            f"[HIGH_TURNOVER] turnover={result.turnover:.3f}. "
            f"MUST increase decay setting by 2. universe=TOP3000 LOCKED. decay minimum=1."
        )

    if "LOW_SHARPE" in failing_checks and result.sharpe < 0.8:
        constraints.append(
            f"[LOW_SHARPE] sharpe={result.sharpe:.3f} < 0.8. "
            "MUST do radical_rewrite of raw_signal layer only. "
            "Use Pattern A-E from playbook. Keep neutralized/smoothed/alpha layers intact."
        )

    if "LOW_SUB_UNIVERSE_SHARPE" in failing_checks:
        constraints.append(
            "[LOW_SUB_UNIVERSE_SHARPE] "
            "MUST change neutralization to INDUSTRY or SUBINDUSTRY. "
            "Do not change expression or universe."
        )

    if "CONCENTRATED_WEIGHT" in failing_checks:
        constraints.append(
            "[CONCENTRATED_WEIGHT] "
            "MUST lower truncation to 0.02 or wrap in group_rank(signal, industry)."
        )

    if "UNITS" in failing_checks:
        constraints.append(
            "[UNITS] Units mismatch. "
            "MUST normalize all intermediate signals using rank() or zscore() "
            "before combining different data fields."
        )

    if not constraints:
        return ""

    header = "=== ACTIVE CONSTRAINTS — follow exactly, these override general guidance ===\n"
    body   = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(constraints))
    footer = "\n=========================================================================\n\n"
    return header + body + footer


# ─────────────────────────────────────────────────────────────────────────────
#  Reward function
# ─────────────────────────────────────────────────────────────────────────────

class RewardCalculator:
    """
    Multi-objective shaped reward.
    confidence now actively weights the reward (suggestion #3 from Gemini Pro review).
    """

    WEIGHTS = {
        "sharpe":     0.45,
        "checks":     0.25,
        "fitness":    0.15,
        "novelty":    0.05,
        "plateau":   -0.05,
        "completion": 0.05,
    }
    COMPLETION_BONUS = 2.0

    @classmethod
    def compute(
        cls,
        result:      SimResult,
        prev_result: Optional[SimResult],
        state:       AlphaState,
        confidence:  float = 0.5,
    ) -> tuple[float, dict]:
        breakdown = {}

        # Sharpe — log-scaled so improvements get harder near the top
        sharpe_score = math.log1p(max(result.sharpe, 0)) / math.log1p(3.0)
        breakdown["sharpe"] = round(sharpe_score * cls.WEIGHTS["sharpe"], 4)

        # Check improvement delta
        prev_fails = len(prev_result.failing_checks) if prev_result else 99
        curr_fails = len(result.failing_checks)
        check_delta = (prev_fails - curr_fails) / max(prev_fails, 1)
        breakdown["checks"] = round(check_delta * cls.WEIGHTS["checks"], 4)

        # Fitness bonus
        fitness_score = min(result.fitness / 2.0, 1.0) if result.fitness > 0 else 0
        breakdown["fitness"] = round(fitness_score * cls.WEIGHTS["fitness"], 4)

        # Novelty — reward visiting new fingerprints
        fp = result.fingerprint
        novelty = 0.1 if fp not in state.visited_fps else 0.0
        breakdown["novelty"] = round(novelty * cls.WEIGHTS["novelty"], 4)
        state.visited_fps.add(fp)

        # Plateau penalty
        plateau_penalty = min(state.plateau_count * 0.02, 0.15)
        breakdown["plateau"] = round(-plateau_penalty, 4)

        # Terminal completion bonus
        if result.passing and result.sharpe >= 1.0:
            breakdown["completion"] = cls.COMPLETION_BONUS
        else:
            breakdown["completion"] = 0.0

        # Confidence multiplier (suggestion #3)
        # Range 0.6 to 1.0 — low confidence dampens but doesn't zero out reward
        confidence_mult = 0.6 + (0.4 * max(0.0, min(1.0, confidence)))
        total = sum(breakdown.values()) * confidence_mult
        breakdown["confidence_mult"] = round(confidence_mult, 3)

        return round(total, 4), breakdown