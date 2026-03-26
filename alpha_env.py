"""
AlphaEnv: RL-style environment for WorldQuant BRAIN alpha generation.

Treats alpha refinement as a Markov Decision Process:
  - State:  current alpha expression + simulation results
  - Action: mutate expression / change settings
  - Reward: shaped reward from Sharpe, checks, fitness, originality
  - Done:   submissible alpha found OR budget exhausted
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ─────────────────────────────────────────────────────────────────────────────
#  Domain types
# ─────────────────────────────────────────────────────────────────────────────

class ActionType(str, Enum):
    MUTATE_EXPRESSION  = "mutate_expression"
    CHANGE_SETTINGS    = "change_settings"
    RADICAL_REWRITE    = "radical_rewrite"
    CROSS_BREED        = "cross_breed"     # blend two high-scoring parents
    WAIT               = "wait"            # no-op (used during warm-up)


@dataclass
class AlphaAction:
    action_type: ActionType
    expression: Optional[str] = None
    settings: Optional[dict] = None
    reasoning: Optional[str] = None
    parent_ids: Optional[list[str]] = None   # for cross-breed


@dataclass
class SimResult:
    """Parsed in-sample simulation result from BRAIN."""
    alpha_id:   str
    sharpe:     float
    fitness:    float
    turnover:   float
    returns:    float
    drawdown:   float
    checks:     list[dict]
    raw:        dict = field(default_factory=dict)

    # ── derived ──────────────────────────────────────────────────────────────
    @property
    def passing(self) -> bool:
        """True if all non-cosmetic checks pass."""
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
            c["name"]
            for c in self.checks
            if c["name"] not in skip and c["result"] not in ("PASS", "PENDING")
        ]

    @property
    def fingerprint(self) -> str:
        return hashlib.md5(
            f"{self.sharpe:.3f}|{self.fitness:.3f}|{self.turnover:.3f}".encode()
        ).hexdigest()[:8]


@dataclass
class AlphaObservation:
    """What the agent sees after each action."""
    episode_id:        str
    step:              int
    sim_result:        Optional[SimResult]
    reward:            float
    done:              bool
    info:              dict        # richer diagnostic info
    system_message:    str
    history_summary:   str        # compact summary of trajectory so far


@dataclass
class AlphaState:
    """Hidden bookkeeping (not fed to LLM, used by env for reward shaping)."""
    episode_id:    str
    difficulty:    str              # "explore" | "exploit" | "diversify"
    best_sharpe:   float = 0.0
    best_alpha_id: str = ""
    step_count:    int = 0
    sim_budget:    int = 20         # max simulations per episode
    plateau_count: int = 0          # steps without improvement
    visited_fps:   set = field(default_factory=set)   # fingerprints
    history:       list[dict] = field(default_factory=list)
    target_achieved: bool = False


# ─────────────────────────────────────────────────────────────────────────────
#  Reward function
# ─────────────────────────────────────────────────────────────────────────────

class RewardCalculator:
    """
    Multi-objective shaped reward.

    Components:
      1. sharpe_reward      – primary signal, non-linear (log-scaled)
      2. check_reward       – +bonus for every check fixed; -penalty per new fail
      3. fitness_bonus      – rewards high fitness (Sharpe × sqrt(returns/turnover))
      4. novelty_bonus      – small reward for exploring distinct regions
      5. plateau_penalty    – discourages stagnation
      6. completion_bonus   – large terminal reward for submissible alpha
    """

    WEIGHTS = {
        "sharpe":      0.45,
        "checks":      0.25,
        "fitness":     0.15,
        "novelty":     0.05,
        "plateau":    -0.05,   # penalty weight (always subtracted)
        "completion":  0.05,   # fraction of completion bonus per step
    }

    COMPLETION_BONUS = 2.0
    CRASH_PENALTY    = -1.0

    @classmethod
    def compute(
        cls,
        result: SimResult,
        prev_result: Optional[SimResult],
        state: AlphaState,
    ) -> tuple[float, dict]:
        breakdown = {}

        # 1. Sharpe – log-scaled so gains become harder near the top
        sharpe_score = math.log1p(max(result.sharpe, 0)) / math.log1p(3.0)
        breakdown["sharpe"] = round(sharpe_score * cls.WEIGHTS["sharpe"], 4)

        # 2. Check improvement delta
        prev_fails = len(prev_result.failing_checks) if prev_result else 99
        curr_fails = len(result.failing_checks)
        check_delta = (prev_fails - curr_fails) / max(prev_fails, 1)
        breakdown["checks"] = round(check_delta * cls.WEIGHTS["checks"], 4)

        # 3. Fitness bonus (only meaningful if > 1.0)
        fitness_score = min(result.fitness / 2.0, 1.0) if result.fitness > 0 else 0
        breakdown["fitness"] = round(fitness_score * cls.WEIGHTS["fitness"], 4)

        # 4. Novelty – reward unique fingerprints
        fp = result.fingerprint
        novelty = 0.1 if fp not in state.visited_fps else 0.0
        breakdown["novelty"] = round(novelty * cls.WEIGHTS["novelty"], 4)
        state.visited_fps.add(fp)

        # 5. Plateau penalty
        plateau_penalty = min(state.plateau_count * 0.02, 0.15)
        breakdown["plateau"] = round(-plateau_penalty, 4)

        # 6. Terminal completion bonus
        if result.passing and result.sharpe >= 1.0:
            breakdown["completion"] = cls.COMPLETION_BONUS
        else:
            breakdown["completion"] = 0.0

        total = sum(breakdown.values())
        return round(total, 4), breakdown