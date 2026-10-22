"""Uniform-bin discretiser: [dx, dy, vy, y] -> single integer state index.

Values outside a feature's range are clipped into the first/last bin. Ranges come
from rollouts of a random and a hand-written policy (see PROGRESS.md):
vy takes the 20 integer values -9..10, so 20 vy bins give an exact encoding.
"""
from dataclasses import dataclass

import numpy as np

# (low, high) per feature, in the order of src.features.FEATURE_NAMES
FEATURE_RANGES = ((-86.0, 198.0), (-105.0, 105.0), (-9.0, 11.0), (0.0, 404.0))

# Bins per feature (dx, dy, vy, y). The selected Q-learning set collapses y to one bin.
BINS_CORE_COARSE = (6, 7, 5, 1)  # 210 states   (resolution ablation)
BINS_CORE = (8, 9, 10, 1)  # 720 states   (selected by tuning)
BINS_CORE_FINE = (12, 15, 20, 1)  # 3,600 states (resolution ablation)
BINS_Y3 = (8, 9, 10, 3)  # 2,160 states (y-resolution ablation)
BINS_COARSE = (8, 9, 10, 6)  # 4,320 states (y-resolution ablation)
BINS_DEFAULT = (12, 15, 20, 10)  # 36,000 states (original design; too sparse)
BINS_FINE = (20, 25, 20, 15)  # 150,000 states


@dataclass(frozen=True)
class Discretizer:
    bins: tuple[int, ...] = BINS_DEFAULT
    ranges: tuple[tuple[float, float], ...] = FEATURE_RANGES

    def __post_init__(self):
        if len(self.bins) != len(self.ranges):
            raise ValueError("bins and ranges must have the same length")
        if any(n < 1 for n in self.bins):
            raise ValueError("every feature needs at least one bin")
        if any(lo >= hi for lo, hi in self.ranges):
            raise ValueError("each range must satisfy low < high")

    @property
    def n_states(self) -> int:
        return int(np.prod(self.bins))

    def bin_indices(self, features: np.ndarray) -> tuple[int, ...]:
        features = np.asarray(features, dtype=np.float64)
        if features.shape != (len(self.bins),):
            raise ValueError(f"expected features of shape ({len(self.bins)},), got {features.shape}")
        idx = []
        for value, n, (lo, hi) in zip(features, self.bins, self.ranges):
            i = int(np.floor((value - lo) / (hi - lo) * n))
            idx.append(min(max(i, 0), n - 1))
        return tuple(idx)

    def encode(self, features: np.ndarray) -> int:
        return int(np.ravel_multi_index(self.bin_indices(features), self.bins))
