"""
Population: Elite archive with confidence-weighted tournament selection.

Changes:
  - AlphaCandidate stores confidence score
  - tournament_select weighted by reward * confidence (suggestion #3)
  - Expression-aware fingerprint deduplication (suggestion #4)
"""

from __future__ import annotations

import heapq
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(order=True)
class AlphaCandidate:
    neg_reward:  float
    alpha_id:    str   = field(compare=False)
    expression:  str   = field(compare=False)
    settings:    dict  = field(compare=False)
    reward:      float = field(compare=False)
    sharpe:      float = field(compare=False)
    fitness:     float = field(compare=False)
    turnover:    float = field(compare=False)
    passing:     bool  = field(compare=False)
    fingerprint: str   = field(compare=False)
    confidence:  float = field(compare=False, default=0.5)
    timestamp:   float = field(compare=False, default_factory=time.time)
    episode_id:  str   = field(compare=False, default="")
    lineage:     list  = field(compare=False, default_factory=list)


class Population:

    def __init__(self, capacity: int = 50):
        self.capacity     = capacity
        self._heap:        list[AlphaCandidate] = []
        self._fingerprints: set[str]            = set()
        self._by_id:       dict[str, AlphaCandidate] = {}

    def try_add(self, candidate: AlphaCandidate) -> bool:
        # Reject duplicates using expression-aware fingerprint
        if candidate.fingerprint in self._fingerprints:
            return False

        if len(self._heap) >= self.capacity:
            worst = self._heap[0]
            if candidate.reward <= worst.reward:
                return False
            heapq.heappop(self._heap)
            self._fingerprints.discard(worst.fingerprint)
            self._by_id.pop(worst.alpha_id, None)

        heapq.heappush(self._heap, candidate)
        self._fingerprints.add(candidate.fingerprint)
        self._by_id[candidate.alpha_id] = candidate
        return True

    def best(self, n: int = 1) -> list[AlphaCandidate]:
        return sorted(self._heap, key=lambda c: -c.reward)[:n]

    def tournament_select(self, k: int = 5) -> Optional[AlphaCandidate]:
        """
        Confidence-weighted tournament selection (suggestion #3).
        Score = reward * confidence so high-confidence elites win more often.
        """
        if not self._heap:
            return None
        pool = random.sample(self._heap, min(k, len(self._heap)))
        return max(pool, key=lambda c: c.reward * max(c.confidence, 0.1))

    def select_diverse_pair(self) -> Optional[tuple[AlphaCandidate, AlphaCandidate]]:
        """Pick top candidate + a random partner for cross-breeding."""
        if len(self._heap) < 2:
            return None
        top  = self.best(1)[0]
        rest = [c for c in self._heap if c.alpha_id != top.alpha_id]
        if not rest:
            return None
        return top, random.choice(rest)

    def should_force_explore(self, threshold: float = 0.5) -> bool:
        """
        Returns True if recent additions are low-confidence,
        suggesting the agent is guessing and exploration is better.
        Implements confidence-based mode switching (suggestion #3).
        """
        recent = sorted(self._heap, key=lambda c: -c.timestamp)[:5]
        if not recent:
            return False
        avg_confidence = sum(c.confidence for c in recent) / len(recent)
        return avg_confidence < threshold

    def statistics(self) -> dict:
        if not self._heap:
            return {
                "size": 0, "best_sharpe": 0.0, "best_reward": 0.0,
                "passing_count": 0, "mean_reward": 0.0, "diversity": 0,
                "mean_confidence": 0.0,
            }
        rewards     = [c.reward for c in self._heap]
        confidences = [c.confidence for c in self._heap]
        return {
            "size":             len(self._heap),
            "best_sharpe":      max(c.sharpe for c in self._heap),
            "best_reward":      max(rewards),
            "mean_reward":      sum(rewards) / len(rewards),
            "passing_count":    sum(1 for c in self._heap if c.passing),
            "diversity":        len(self._fingerprints),
            "mean_confidence":  round(sum(confidences) / len(confidences), 3),
        }

    def save(self, path: str):
        data = []
        for c in sorted(self._heap, key=lambda c: -c.reward):
            data.append({
                "alpha_id":    c.alpha_id,
                "expression":  c.expression,
                "settings":    c.settings,
                "reward":      c.reward,
                "sharpe":      c.sharpe,
                "fitness":     c.fitness,
                "turnover":    c.turnover,
                "passing":     c.passing,
                "fingerprint": c.fingerprint,
                "confidence":  c.confidence,
                "timestamp":   c.timestamp,
                "episode_id":  c.episode_id,
                "lineage":     c.lineage,
            })
        Path(path).write_text(json.dumps(data, indent=2))
        print(f"[Population] Saved {len(data)} candidates to {path}")

    def load(self, path: str):
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
                confidence  = row.get("confidence", 0.5),
                timestamp   = row.get("timestamp", 0.0),
                episode_id  = row.get("episode_id", ""),
                lineage     = row.get("lineage", []),
            )
            self.try_add(c)
        print(f"[Population] Loaded {len(self._heap)} candidates from {path}")