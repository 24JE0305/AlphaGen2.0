"""
Population: Maintains a ranked population of alpha candidates across episodes.

This is the key architectural upgrade over the naive refine.py single-threaded approach.

Features:
  - Elite archive: keeps the top-K alphas ever found (by reward)
  - Diversity enforcement: won't add near-duplicate fingerprints
  - Parent selection: tournament selection for cross-breeding
  - Statistics tracking: convergence metrics for adaptive mode switching
"""

from __future__ import annotations

import heapq
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(order=True)
class AlphaCandidate:
    """A single candidate in the population."""
    # order=True means Python will sort by the first field
    neg_reward: float                    # negated for min-heap (want max reward)
    alpha_id:   str   = field(compare=False)
    expression: str   = field(compare=False)
    settings:   dict  = field(compare=False)
    reward:     float = field(compare=False)
    sharpe:     float = field(compare=False)
    fitness:    float = field(compare=False)
    turnover:   float = field(compare=False)
    passing:    bool  = field(compare=False)
    fingerprint: str  = field(compare=False)
    timestamp:  float = field(compare=False, default_factory=time.time)
    episode_id: str   = field(compare=False, default="")
    lineage:    list  = field(compare=False, default_factory=list)  # parent alpha_ids


class Population:
    """
    Elite archive with diversity enforcement.

    Uses a min-heap so we can efficiently evict the worst candidate
    when capacity is exceeded.
    """

    def __init__(self, capacity: int = 50, diversity_threshold: float = 0.05):
        self.capacity = capacity
        self.diversity_threshold = diversity_threshold
        self._heap: list[AlphaCandidate] = []
        self._fingerprints: set[str] = set()
        self._by_id: dict[str, AlphaCandidate] = {}

    def try_add(self, candidate: AlphaCandidate) -> bool:
        """
        Add candidate if it improves the population.
        Returns True if added.
        """
        # Reject near-duplicates (same performance fingerprint)
        if candidate.fingerprint in self._fingerprints:
            return False

        # If at capacity, only add if better than worst
        if len(self._heap) >= self.capacity:
            worst = self._heap[0]  # min-heap → worst is first
            if candidate.reward <= worst.reward:
                return False
            # Evict the worst
            heapq.heappop(self._heap)
            self._fingerprints.discard(worst.fingerprint)
            del self._by_id[worst.alpha_id]

        heapq.heappush(self._heap, candidate)
        self._fingerprints.add(candidate.fingerprint)
        self._by_id[candidate.alpha_id] = candidate
        return True

    def best(self, n: int = 1) -> list[AlphaCandidate]:
        """Return top-n candidates by reward."""
        sorted_heap = sorted(self._heap, key=lambda c: -c.reward)
        return sorted_heap[:n]

    def tournament_select(self, k: int = 5) -> Optional[AlphaCandidate]:
        """Select a parent via tournament selection (good for diversity)."""
        import random
        if not self._heap:
            return None
        candidates = random.sample(self._heap, min(k, len(self._heap)))
        return max(candidates, key=lambda c: c.reward)

    def select_diverse_pair(self) -> Optional[tuple[AlphaCandidate, AlphaCandidate]]:
        """Select two candidates that are most different from each other."""
        if len(self._heap) < 2:
            return None
        # Simple heuristic: pick best + random other
        import random
        top = self.best(1)[0]
        rest = [c for c in self._heap if c.alpha_id != top.alpha_id]
        if not rest:
            return None
        partner = random.choice(rest)
        return top, partner

    def statistics(self) -> dict:
        if not self._heap:
            return {"size": 0, "best_sharpe": 0.0, "best_reward": 0.0,
                    "passing_count": 0, "mean_reward": 0.0}
        rewards = [c.reward for c in self._heap]
        return {
            "size":          len(self._heap),
            "best_sharpe":   max(c.sharpe for c in self._heap),
            "best_reward":   max(rewards),
            "mean_reward":   sum(rewards) / len(rewards),
            "passing_count": sum(1 for c in self._heap if c.passing),
            "diversity":     len(self._fingerprints),
        }

    def save(self, path: str):
        """Persist population to JSON for resumability."""
        data = []
        for c in sorted(self._heap, key=lambda c: -c.reward):
            row = {
                "alpha_id":    c.alpha_id,
                "expression":  c.expression,
                "settings":    c.settings,
                "reward":      c.reward,
                "sharpe":      c.sharpe,
                "fitness":     c.fitness,
                "turnover":    c.turnover,
                "passing":     c.passing,
                "fingerprint": c.fingerprint,
                "timestamp":   c.timestamp,
                "episode_id":  c.episode_id,
                "lineage":     c.lineage,
            }
            data.append(row)
        Path(path).write_text(json.dumps(data, indent=2))
        print(f"[Population] Saved {len(data)} candidates to {path}")

    def load(self, path: str):
        """Reload population from JSON."""
        if not Path(path).exists():
            return
        data = json.loads(Path(path).read_text())
        for row in data:
            c = AlphaCandidate(
                neg_reward  = -row["reward"],
                alpha_id    = row["alpha_id"],
                expression  = row["expression"],
                settings    = row["settings"],
                reward      = row["reward"],
                sharpe      = row["sharpe"],
                fitness     = row["fitness"],
                turnover    = row["turnover"],
                passing     = row["passing"],
                fingerprint = row["fingerprint"],
                timestamp   = row.get("timestamp", 0.0),
                episode_id  = row.get("episode_id", ""),
                lineage     = row.get("lineage", []),
            )
            self.try_add(c)
        print(f"[Population] Loaded {len(self._heap)} candidates from {path}")