import numpy as np

from scripts.play import GROUND_Y, classify_death, frame_window, pick_episode, rollout
from src.agents.baselines import HeuristicPolicy, RandomPolicy
from src.env import make_env
from src.features import PLAYER_X


def _obs(pipe_x, gap_top, gap_bottom, y):
    """Target pipe first, two sentinels after it; vy and rotation are irrelevant here."""
    return np.array([pipe_x, gap_top, gap_bottom, 288, 0, 512, 288, 0, 512, y, 0, 0], dtype=np.int64)


def test_ground_constant_matches_environment():
    env = make_env(None)
    env.reset(seed=0)
    assert env.unwrapped._ground["y"] == GROUND_Y
    env.close()


def test_classify_death_by_region():
    assert classify_death(_obs(200, 100, 200, 390)) == "ground"  # ground wins even far from any pipe
    assert classify_death(_obs(PLAYER_X, 100, 200, 60)) == "pipe_top"  # overlapping, above the gap
    assert classify_death(_obs(PLAYER_X, 100, 200, 190)) == "pipe_bottom"  # overlapping, bird bottom below the gap
    assert classify_death(_obs(PLAYER_X, 100, 200, 120)) == "other"  # inside the gap: not a pipe crash
    assert classify_death(_obs(200, 100, 200, 60)) == "other"  # pipe still ahead


def test_every_crash_of_real_rollouts_is_attributed():
    for policy in (HeuristicPolicy(), RandomPolicy(0.5, seed=0), RandomPolicy(0.1, seed=0)):
        records, _ = rollout(policy, 15, seed=3)
        assert all(r["cause"] in ("ground", "pipe_top", "pipe_bottom") for r in records if not r["capped"])


def test_rollout_is_deterministic_and_rendering_does_not_change_it():
    plain, _ = rollout(HeuristicPolicy(), 4, seed=5)
    rendered, frames = rollout(HeuristicPolicy(), 4, seed=5, render=(2, 0, 9))
    assert plain == rendered
    assert len(frames) == 11 and frames[0].shape == (512, 288, 3)  # 10 steps + the final frame


def test_pick_episode_and_frame_window():
    records = [{"capped": False, "episode": 0}, {"capped": True, "episode": 1}]
    assert pick_episode(records, "death")["episode"] == 0
    assert pick_episode(records, "success")["episode"] == 1
    assert pick_episode(records[:1], "success") is None
    assert frame_window(5000, 900, "death") == (4100, 5000)
    assert frame_window(300, 900, "death") == (0, 300)
    assert frame_window(5000, 900, "success") == (0, 899)
