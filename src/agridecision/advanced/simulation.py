"""Sequential market-selection simulator using an upper-confidence-bound policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class UCBBandit:
    market_names: list[str]
    exploration: float = 2.0

    def __post_init__(self) -> None:
        self.counts = np.zeros(len(self.market_names), dtype=int)
        self.mean_rewards = np.zeros(len(self.market_names), dtype=float)

    def choose(self, step: int) -> int:
        unexplored = np.flatnonzero(self.counts == 0)
        if len(unexplored):
            return int(unexplored[0])
        bonus = np.sqrt(self.exploration * np.log(step + 1) / self.counts)
        return int(np.argmax(self.mean_rewards + bonus))

    def update(self, action: int, reward: float) -> None:
        self.counts[action] += 1
        count = self.counts[action]
        self.mean_rewards[action] += (reward - self.mean_rewards[action]) / count


def simulate_market_policy(
    true_mean_rewards: dict[str, float],
    *,
    reward_std: float = 80.0,
    steps: int = 365,
    random_state: int = 42,
) -> tuple[pd.DataFrame, UCBBandit]:
    rng = np.random.default_rng(random_state)
    names = list(true_mean_rewards)
    means = np.asarray([true_mean_rewards[name] for name in names], dtype=float)
    bandit = UCBBandit(names)
    optimal = float(means.max())
    rows: list[dict[str, object]] = []
    cumulative_regret = 0.0
    for step in range(steps):
        action = bandit.choose(step)
        reward = float(rng.normal(means[action], reward_std))
        bandit.update(action, reward)
        cumulative_regret += optimal - means[action]
        rows.append(
            {
                "step": step + 1,
                "chosen_market": names[action],
                "reward": reward,
                "cumulative_regret": cumulative_regret,
            }
        )
    return pd.DataFrame(rows), bandit
