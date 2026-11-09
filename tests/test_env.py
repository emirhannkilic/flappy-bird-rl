import pytest

from src.agents.baselines import HeuristicPolicy
from src.env import NoAliveReward, make_env
from src.features import extract_features


@pytest.mark.parametrize("raw, expected", [(0.1, 0.0), (-0.5, -0.5), (1.0, 1.0), (-1.0, -1.0)])
def test_no_alive_reward_maps_each_reward_separately(raw, expected):
    assert NoAliveReward(make_env()).reward(raw) == expected


def test_no_alive_reward_matches_raw_env_except_alive_steps():
    """Same seed + same actions: wrapped rewards equal raw rewards with 0.1 -> 0 and nothing else.
    Two rollouts together produce every reward type: 30 opening flaps hit the ceiling (-0.5, then
    death), the plain heuristic collects alive / pipe / death."""
    policy, raw_seen = HeuristicPolicy(), set()
    for opening_flaps in (0, 30):
        raw_env, wrapped_env = make_env(score_limit=None), make_env(score_limit=None, alive_reward=False)
        obs, _ = raw_env.reset(seed=3)
        wrapped_env.reset(seed=3)
        for step in range(5000):
            action = 1 if step < opening_flaps else policy(extract_features(obs))
            obs, r_raw, term, trunc, _ = raw_env.step(action)
            _, r_wrapped, term_w, trunc_w, _ = wrapped_env.step(action)
            raw_seen.add(r_raw)
            assert r_wrapped == (0.0 if r_raw == 0.1 else r_raw)
            assert (term, trunc) == (term_w, trunc_w)
            if term or trunc:
                break
    assert {0.1, -0.5, 1.0, -1.0} <= raw_seen
