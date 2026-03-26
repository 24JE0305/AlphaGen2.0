"""
AlphaEnvironment: RL environment wrapping WorldQuant BRAIN.

Changes in this version:
  - Hard constraints enforced before every simulation
  - expression stored in SimResult for expression-aware fingerprinting
  - CHANGE_SETTINGS guardrail: logs warning if LLM leaks expression change
  - confidence passed through to RewardCalculator
  - mode starts correctly based on seed quality
"""

from __future__ import annotations

import uuid
from typing import Optional

from alpha_env import (
    ActionType,
    AlphaAction,
    AlphaObservation,
    AlphaState,
    RewardCalculator,
    SimResult,
    enforce_hard_constraints,
)


def _safe_float(val, default=0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


class AlphaEnvironment:

    PLATEAU_THRESHOLD  = 3
    DIVERSIFY_INTERVAL = 6

    def __init__(self, brain_session, sim_budget: int = 20):
        self.brain_session   = brain_session
        self.sim_budget      = sim_budget
        self._state:          Optional[AlphaState]  = None
        self._current_expression: str               = ""
        self._current_settings:   dict              = {}
        self._prev_result:    Optional[SimResult]   = None
        self._current_result: Optional[SimResult]   = None
        self._mode:           str                   = "explore"

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(
        self,
        seed_expression: str,
        settings:        dict,
        difficulty:      str = "explore",
        episode_id:      Optional[str] = None,
    ) -> AlphaObservation:

        self._state = AlphaState(
            episode_id = episode_id or str(uuid.uuid4()),
            difficulty = difficulty,
            sim_budget = self.sim_budget,
        )
        self._current_expression = seed_expression
        self._current_settings   = enforce_hard_constraints(dict(settings))
        self._prev_result        = None
        self._current_result     = None
        self._mode               = "explore"

        result = self._simulate(seed_expression, self._current_settings)
        if result:
            self._current_result     = result
            self._state.best_sharpe  = result.sharpe
            self._state.best_alpha_id = result.alpha_id
            reward, breakdown = RewardCalculator.compute(result, None, self._state)
            # Start in correct mode based on seed quality
            self._mode = "exploit" if result.sharpe >= 1.2 else "explore"
        else:
            reward, breakdown = 0.0, {}

        return AlphaObservation(
            episode_id     = self._state.episode_id,
            step           = 0,
            sim_result     = result,
            reward         = reward,
            done           = False,
            info           = {"mode": self._mode, "reward_breakdown": breakdown},
            system_message = f"Seed sharpe={result.sharpe:.3f}, mode={self._mode}" if result else "Seed sim failed.",
            history_summary= "Step 0: Baseline established.",
        )

    def step(self, action: AlphaAction) -> AlphaObservation:
        if not self._state:
            raise RuntimeError("Call reset() before step()")

        self._state.step_count += 1
        step = self._state.step_count

        # ── CHANGE_SETTINGS guardrail (suggestion #5 from Gemini Pro review) ──
        if action.action_type == ActionType.CHANGE_SETTINGS:
            new_expression = self._current_expression   # lock expression
            # Warn if LLM leaked an expression change anyway
            if (action.expression and
                action.expression.strip() != self._current_expression.strip()):
                print(
                    f"[Guardrail] action=change_settings but expression changed. "
                    f"Expression locked. New settings will be applied."
                )
        else:
            new_expression = action.expression or self._current_expression

        # Apply settings and enforce hard constraints
        new_settings = dict(self._current_settings)
        if action.settings:
            new_settings.update(action.settings)
        new_settings = enforce_hard_constraints(new_settings)

        # Simulate
        result = self._simulate(new_expression, new_settings)
        self._prev_result = self._current_result
        message_parts = []
        improved = False

        if result is None:
            reward    = -0.3
            breakdown = {"error_penalty": -0.3}
            message_parts.append("Simulation failed — expression may be invalid.")
            done = step >= self._state.sim_budget
        else:
            self._current_result     = result
            self._current_expression = new_expression
            self._current_settings   = new_settings

            reward, breakdown = RewardCalculator.compute(
                result, self._prev_result, self._state,
                confidence=action.confidence,
            )

            improved = result.sharpe > self._state.best_sharpe
            if improved:
                self._state.best_sharpe    = result.sharpe
                self._state.best_alpha_id  = result.alpha_id
                self._state.plateau_count  = 0
                message_parts.append(f"New best! Sharpe={result.sharpe:.3f}")
            else:
                self._state.plateau_count += 1

            if result.failing_checks:
                message_parts.append(f"Failing: {', '.join(result.failing_checks)}")
            else:
                message_parts.append("All checks passing")

            done = (
                (result.passing and result.sharpe >= 1.0)
                or step >= self._state.sim_budget
            )
            if result.passing and result.sharpe >= 1.0:
                self._state.target_achieved = True
                message_parts.append("Submissible alpha found!")

        self._update_mode(step, result, action.confidence)

        self._state.history.append({
            "step":       step,
            "action":     action.action_type,
            "reward":     reward,
            "sharpe":     result.sharpe if result else 0.0,
            "fitness":    result.fitness if result else 0.0,
            "turnover":   result.turnover if result else 0.0,
            "passing":    result.passing if result else False,
            "mode":       self._mode,
            "confidence": action.confidence,
        })

        obs = AlphaObservation(
            episode_id     = self._state.episode_id,
            step           = step,
            sim_result     = result,
            reward         = reward,
            done           = done,
            info           = {
                "mode":            self._mode,
                "reward_breakdown":breakdown,
                "improved":        improved,
                "plateau_count":   self._state.plateau_count,
                "best_sharpe":     self._state.best_sharpe,
                "sim_budget_left": self._state.sim_budget - step,
            },
            system_message  = " | ".join(message_parts),
            history_summary = self._build_history_summary(),
        )
        return obs

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def state(self) -> Optional[AlphaState]:
        return self._state

    @property
    def current_expression(self) -> str:
        return self._current_expression

    @property
    def current_settings(self) -> dict:
        return self._current_settings

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _simulate(self, expression: str, settings: dict) -> Optional[SimResult]:
        from ace_extensions import simulate_single_alpha
        from alpha_utils import fix_fastexpr

        clean_expr    = fix_fastexpr(expression)
        alpha_payload = self._build_alpha_payload(clean_expr, settings)

        try:
            sim = simulate_single_alpha(self.brain_session, alpha_payload)
            if not sim or not sim.get("id"):
                return None
            result = self._parse_sim_result(sim)
            result.expression = clean_expr   # store for fingerprinting
            return result
        except Exception as e:
            print(f"[Env] Simulation error: {e}")
            return None

    def _parse_sim_result(self, sim: dict) -> SimResult:
        is_data  = sim.get("is", {})
        return SimResult(
            alpha_id = sim.get("id", "unknown"),
            sharpe   = _safe_float(is_data.get("sharpe")),
            fitness  = _safe_float(is_data.get("fitness")),
            turnover = _safe_float(is_data.get("turnover")),
            returns  = _safe_float(is_data.get("returns")),
            drawdown = _safe_float(is_data.get("drawdown")),
            checks   = is_data.get("checks", []),
            raw      = sim,
        )

    def _build_alpha_payload(self, expression: str, settings: dict) -> dict:
        return {
            "type": "REGULAR",
            "settings": {
                "instrumentType": "EQUITY",
                "region":         settings.get("region", "USA"),
                "universe":       settings.get("universe", "TOP3000"),
                "delay":          settings.get("delay", 1),
                "decay":          max(int(settings.get("decay", 4)), 1),
                "neutralization": settings.get("neutralization", "SUBINDUSTRY"),
                "truncation":     settings.get("truncation", 0.05),
                "pasteurization": "ON",
                "unitHandling":   "VERIFY",
                "nanHandling":    "ON",
                "language":       "FASTEXPR",
                "visualization":  False,
            },
            "regular": expression,
        }

    def _update_mode(self, step: int, result: Optional[SimResult], confidence: float = 0.5):
        """
        Adaptive mode switching.
        Confidence score now influences mode (suggestion #3):
        low confidence during exploit -> force explore instead.
        """
        if step % self.DIVERSIFY_INTERVAL == 0 and step > 0:
            self._mode = "diversify"
            return

        if result is None:
            self._mode = "explore"
            return

        if result.passing and result.sharpe >= 1.0:
            self._mode = "exploit"
        elif result.sharpe > 1.2:
            # Good signal but low confidence -> diversify for fresh ideas
            if confidence < 0.5:
                self._mode = "diversify"
            else:
                self._mode = "exploit"
        elif self._state.plateau_count >= self.PLATEAU_THRESHOLD:
            self._mode = "explore"
            self._state.plateau_count = 0
        elif result.sharpe < 0.5:
            self._mode = "explore"
        else:
            # In the middle zone: use confidence to decide
            self._mode = "exploit" if confidence >= 0.6 else "explore"

    def _build_history_summary(self) -> str:
        if not self._state or not self._state.history:
            return "No history yet."
        lines = []
        for e in self._state.history[-5:]:
            status = "PASS" if e["passing"] else "FAIL"
            lines.append(
                f"Step {e['step']:02d} [{e['mode']}] {e['action']} | "
                f"sharpe={e['sharpe']:.3f} fitness={e['fitness']:.3f} "
                f"turn={e['turnover']:.3f} reward={e['reward']:+.3f} conf={e['confidence']:.2f} {status}"
            )
        return "\n".join(lines)

    def build_sim_summary(self, result: Optional[SimResult]) -> str:
        if result is None:
            return "No simulation result yet."
        lines = [
            f"Alpha ID:  {result.alpha_id}",
            f"Sharpe:    {result.sharpe:.4f}",
            f"Fitness:   {result.fitness:.4f}",
            f"Turnover:  {result.turnover:.4f}",
            f"Returns:   {result.returns:.4f}",
            f"Drawdown:  {result.drawdown:.4f}",
            f"Passing:   {result.passing}",
        ]
        if result.failing_checks:
            lines.append(f"Failing:   {', '.join(result.failing_checks)}")
        else:
            lines.append("Checks:    ALL PASSING")
        return "\n".join(lines)