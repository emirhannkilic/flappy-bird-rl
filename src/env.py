"""Environment factory shared by training, evaluation and baselines."""
import gymnasium as gym

import flappy_bird_gymnasium  # noqa: F401  (registers FlappyBird-v0)


def make_env(score_limit: int | None = 100) -> gym.Env:
    """Raw-pixel 12-dim observation; `truncated` fires when `score_limit` pipes are passed."""
    return gym.make(
        "FlappyBird-v0",
        use_lidar=False,
        normalize_obs=False,
        score_limit=score_limit,
        disable_env_checker=True,  # obs is int64 while the declared space is float64
    )
