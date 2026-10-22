import gymnasium as gym
import numpy as np
import pytest

import flappy_bird_gymnasium  # noqa: F401  (registers FlappyBird-v0)
from src.features import (
    PLAYER_RIGHT,
    PLAYER_X,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    extract_features,
    select_target_pipe,
)

SENTINEL = (SCREEN_WIDTH, 0, SCREEN_HEIGHT)


def make_obs(pipes, y=200, vy=0, rot=0):
    flat = [v for pipe in pipes for v in pipe]
    return np.array(flat + [y, vy, rot], dtype=np.int64)


def test_constants_match_environment():
    env = gym.make("FlappyBird-v0", use_lidar=False, normalize_obs=False).unwrapped
    env.reset(seed=0)
    assert env._screen_width == SCREEN_WIDTH
    assert env._screen_height == SCREEN_HEIGHT
    assert env._player_x == PLAYER_X


def test_pipe_ahead_is_target():
    obs = make_obs([(150, 100, 200), (294 - 6, 150, 250), SENTINEL])
    assert select_target_pipe(obs) == (150.0, 100.0, 200.0)


def test_overlapping_pipe_stays_target_after_midpoint_passed():
    # x=20 -> pipe mid (46) is already behind the bird mid (74), but right edge 72 > 57.
    obs = make_obs([(20, 100, 200), (170, 150, 250), SENTINEL])
    assert select_target_pipe(obs)[0] == 20.0


def test_cleared_pipe_is_skipped():
    # x=5 -> right edge 57 is not > PLAYER_X, so the bird has cleared it.
    obs = make_obs([(5, 100, 200), (155, 150, 250), SENTINEL])
    assert select_target_pipe(obs) == (155.0, 150.0, 250.0)


def test_sentinels_are_never_selected_when_real_pipe_exists():
    obs = make_obs([SENTINEL, (150, 100, 200), SENTINEL])
    assert select_target_pipe(obs) == (150.0, 100.0, 200.0)


def test_only_sentinels_falls_back_to_screen_middle():
    obs = make_obs([SENTINEL, SENTINEL, SENTINEL], y=244)
    dx, dy, _, _ = extract_features(obs)
    assert dx == SCREEN_WIDTH - PLAYER_RIGHT
    assert dy == pytest.approx(244 + 12 - SCREEN_HEIGHT / 2)


def test_feature_values_and_sign_convention():
    obs = make_obs([(200, 160, 260), SENTINEL, SENTINEL], y=300, vy=5)
    dx, dy, vy, y = extract_features(obs)
    assert dx == 200 - PLAYER_RIGHT
    assert dy == 300 + 12 - 210  # bird below gap centre -> positive
    assert (vy, y) == (5, 300)
    above = extract_features(make_obs([(200, 160, 260), SENTINEL, SENTINEL], y=100))
    assert above[1] < 0


def test_output_is_float64_and_shape_checked():
    out = extract_features(make_obs([(200, 160, 260), SENTINEL, SENTINEL]))
    assert out.dtype == np.float64 and out.shape == (4,)
    with pytest.raises(ValueError):
        extract_features(np.zeros(180))


def test_features_finite_along_random_rollout():
    env = gym.make("FlappyBird-v0", use_lidar=False, normalize_obs=False, score_limit=100)
    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed=0)
    for _ in range(500):
        feats = extract_features(obs)
        assert np.isfinite(feats).all()
        obs, _, terminated, truncated, _ = env.step(int(rng.integers(2)))
        if terminated or truncated:
            obs, _ = env.reset()


def test_target_is_still_the_passed_pipe_when_score_increments():
    """The env scores at the pipe midpoint, while the bird still overlaps the pipe.

    The target must not have switched to the next gap yet (dx < 0). Uses a simple
    heuristic controller so the bird actually passes pipes; asserts that it did.
    """
    env = gym.make("FlappyBird-v0", use_lidar=False, normalize_obs=False, score_limit=5)
    obs, _ = env.reset(seed=0)
    passes = 0
    for _ in range(5000):
        dx, dy, vy, _ = extract_features(obs)
        action = int(dy > 8)  # flap when the bird is below the gap centre
        score_before = env.unwrapped._score
        obs, _, terminated, truncated, info = env.step(action)
        if info["score"] > score_before:
            passes += 1
            assert extract_features(obs)[0] < 0
        if terminated or truncated:
            break
    assert passes >= 1
