"""Environment factory shared by training, evaluation and baselines."""
import gymnasium as gym

import flappy_bird_gymnasium  # noqa: F401  (registers FlappyBird-v0)


class NoAliveReward(gym.RewardWrapper):
    """Zero the +0.1 per-step survival reward; pipe (+1), ceiling (-0.5) and death (-1) are kept."""

    ALIVE = 0.1

    def reward(self, reward):
        return 0.0 if reward == self.ALIVE else reward


def make_env(score_limit: int | None = 100, alive_reward: bool = True, render_mode: str | None = None) -> gym.Env:
    """Raw-pixel 12-dim observation; `truncated` fires when `score_limit` pipes are passed.
    `alive_reward=False` is the reward ablation (training env only; eval scores come from `info`).
    `render_mode="rgb_array"` is for videos only: frames are drawn on explicit `env.render()` calls, never in `step`."""
    env = gym.make(
        "FlappyBird-v0",
        use_lidar=False,
        normalize_obs=False,
        score_limit=score_limit,
        render_mode=render_mode,
        disable_env_checker=True,  # obs is int64 while the declared space is float64
    )
    return env if alive_reward else NoAliveReward(env)
