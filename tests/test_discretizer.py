import itertools

import numpy as np
import pytest

from src.discretizer import (
    BINS_COARSE,
    BINS_DEFAULT,
    BINS_FINE,
    FEATURE_RANGES,
    Discretizer,
)


def test_state_counts_match_documented_sizes():
    assert Discretizer(BINS_COARSE).n_states == 4_320
    assert Discretizer(BINS_DEFAULT).n_states == 36_000
    assert Discretizer(BINS_FINE).n_states == 150_000


def test_range_edges_and_clipping():
    d = Discretizer()
    lows = np.array([lo for lo, _ in FEATURE_RANGES])
    highs = np.array([hi for _, hi in FEATURE_RANGES])
    assert d.bin_indices(lows) == (0, 0, 0, 0)
    assert d.bin_indices(highs) == tuple(n - 1 for n in d.bins)  # high edge clips into last bin
    assert d.bin_indices(lows - 1e6) == (0, 0, 0, 0)
    assert d.bin_indices(highs + 1e6) == tuple(n - 1 for n in d.bins)


def test_vy_default_bins_are_exact_integers():
    d = Discretizer()
    seen = [d.bin_indices(np.array([0, 0, vy, 100]))[2] for vy in range(-9, 11)]
    assert seen == list(range(20))


def test_encode_is_in_range_and_injective_on_bin_centres():
    d = Discretizer(BINS_COARSE)
    centres = [
        lo + (np.arange(n) + 0.5) * (hi - lo) / n for n, (lo, hi) in zip(d.bins, d.ranges)
    ]
    codes = {d.encode(np.array(p)) for p in itertools.product(*centres)}
    assert len(codes) == d.n_states
    assert min(codes) == 0 and max(codes) == d.n_states - 1


def test_encode_is_deterministic_and_returns_int():
    d = Discretizer()
    f = np.array([12.3, -4.5, 3.0, 150.0])
    assert d.encode(f) == d.encode(f.copy())
    assert isinstance(d.encode(f), int)


def test_input_validation():
    with pytest.raises(ValueError):
        Discretizer().encode(np.zeros(3))
    with pytest.raises(ValueError):
        Discretizer(bins=(0, 1, 1, 1))
    with pytest.raises(ValueError):
        Discretizer(bins=(2, 2, 2))
    with pytest.raises(ValueError):
        Discretizer(ranges=((1.0, 1.0),) * 4)
