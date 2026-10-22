"""Feature extraction: 12-dim raw FlappyBird-v0 observation -> 4 agent features.

Observation layout (use_lidar=False, normalize_obs=False): three pipes sorted by
x as (left_x, gap_top, gap_bottom) followed by (player_y, player_vel_y, player_rot).
Pipes that have not entered the screen yet are reported as the sentinel
(SCREEN_WIDTH, 0, SCREEN_HEIGHT).
"""
import numpy as np
from flappy_bird_gymnasium.envs.constants import PIPE_WIDTH, PLAYER_HEIGHT, PLAYER_WIDTH

SCREEN_WIDTH = 288
SCREEN_HEIGHT = 512
PLAYER_X = int(SCREEN_WIDTH * 0.2)  # same formula as the environment
PLAYER_RIGHT = PLAYER_X + PLAYER_WIDTH

N_PIPES = 3
FEATURE_NAMES = ("dx", "dy", "vy", "y")


def _is_sentinel(gap_top: float, gap_bottom: float) -> bool:
    return gap_top == 0 and gap_bottom == SCREEN_HEIGHT


def select_target_pipe(obs: np.ndarray) -> tuple[float, float, float]:
    """Return (left_x, gap_top, gap_bottom) of the pipe that currently constrains the bird.

    That is the first real pipe whose right edge has not yet passed the bird's left edge.
    Falls back to the first sentinel if no real pipe qualifies.
    """
    pipes = np.asarray(obs, dtype=np.float64)[: 3 * N_PIPES].reshape(N_PIPES, 3)
    for x, gap_top, gap_bottom in pipes:
        if _is_sentinel(gap_top, gap_bottom):
            continue
        if x + PIPE_WIDTH > PLAYER_X:
            return float(x), float(gap_top), float(gap_bottom)
    return float(SCREEN_WIDTH), 0.0, float(SCREEN_HEIGHT)


def extract_features(obs: np.ndarray) -> np.ndarray:
    """Return float64 array [dx, dy, vy, y].

    dx: target pipe left edge - bird right edge (negative while overlapping).
    dy: bird centre - gap centre (positive = bird is below the gap centre).
    vy: bird vertical velocity. y: bird absolute vertical position.
    """
    obs = np.asarray(obs, dtype=np.float64)
    if obs.shape != (12,):
        raise ValueError(f"expected observation of shape (12,), got {obs.shape}")
    pipe_x, gap_top, gap_bottom = select_target_pipe(obs)
    y, vy = obs[9], obs[10]
    dx = pipe_x - PLAYER_RIGHT
    dy = (y + PLAYER_HEIGHT / 2) - (gap_top + gap_bottom) / 2
    return np.array([dx, dy, vy, y], dtype=np.float64)
