"""Tabular Q-learning over the discretised feature vector."""
import numpy as np

from src.discretizer import Discretizer

N_ACTIONS = 2


class QLearningAgent:
    def __init__(
        self,
        discretizer: Discretizer | None = None,
        alpha: float = 0.1,
        gamma: float = 0.99,
        seed: int = 0,
        explore_flap_prob: float = 0.5,
        tie_break: str = "random",
    ):
        if tie_break not in ("random", "idle"):
            raise ValueError("tie_break must be 'random' or 'idle'")
        self.discretizer = discretizer or Discretizer()
        self.alpha = alpha
        self.gamma = gamma
        self.explore_flap_prob = explore_flap_prob  # P(flap) when the epsilon branch fires
        self.tie_break = tie_break  # exploitation ties: 'random' or 'idle' (lowest index = no flap)
        self.q = np.zeros((self.discretizer.n_states, N_ACTIONS))
        self.rng = np.random.default_rng(seed)  # exploration and exploration tie-breaks only

    def act(self, features: np.ndarray, epsilon: float) -> int:
        """Biased epsilon-greedy: with prob. epsilon flap w.p. `explore_flap_prob`, else greedy."""
        if self.rng.random() < epsilon:
            return int(self.rng.random() < self.explore_flap_prob)
        row = self.q[self.discretizer.encode(features)]
        if self.tie_break == "idle":
            return int(np.argmax(row))
        return int(self.rng.choice(np.flatnonzero(row == row.max())))

    def greedy(self, features: np.ndarray) -> int:
        """Deterministic evaluation policy (ties -> action 0); does not touch the RNG."""
        return int(np.argmax(self.q[self.discretizer.encode(features)]))

    def observe(self, features, action, reward, next_features, terminated: bool) -> None:
        """Off-policy TD update. Only `terminated` cuts the bootstrap; truncation does not."""
        s = self.discretizer.encode(features)
        target = reward
        if not terminated:
            target += self.gamma * self.q[self.discretizer.encode(next_features)].max()
        self.q[s, action] += self.alpha * (target - self.q[s, action])

    def save(self, path) -> None:
        np.save(path, self.q)

    def load(self, path) -> None:
        q = np.load(path)
        if q.shape != self.q.shape:
            raise ValueError(f"checkpoint shape {q.shape} != agent shape {self.q.shape}")
        self.q = q
