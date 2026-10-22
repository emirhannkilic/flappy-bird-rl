"""Baseline policies: callables mapping the feature vector to an action."""
import numpy as np

FLAP_THRESHOLD = 8.0  # flap when the bird centre is more than this many px below the gap centre


class RandomPolicy:
    def __init__(self, flap_prob: float = 0.1, seed: int = 0):
        self.flap_prob = flap_prob
        self.rng = np.random.default_rng(seed)

    def __call__(self, features: np.ndarray) -> int:
        return int(self.rng.random() < self.flap_prob)


class HeuristicPolicy:
    def __call__(self, features: np.ndarray) -> int:
        return int(features[1] > FLAP_THRESHOLD)
