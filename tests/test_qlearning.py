import numpy as np
import pytest

from src.agents.baselines import HeuristicPolicy, RandomPolicy
from src.agents.qlearning import QLearningAgent

F = np.array([10.0, 0.0, 0.0, 100.0])
F2 = np.array([6.0, 0.0, 0.0, 100.0])


def agent(**kw):
    return QLearningAgent(alpha=0.1, gamma=0.9, seed=0, **kw)


def test_td_update_bootstraps_on_non_terminal():
    a = agent()
    a.q[a.discretizer.encode(F2)] = [2.0, 1.0]
    a.observe(F, 1, 1.0, F2, terminated=False)
    assert a.q[a.discretizer.encode(F), 1] == pytest.approx(0.1 * (1.0 + 0.9 * 2.0))


def test_terminal_transition_does_not_bootstrap():
    a = agent()
    a.q[a.discretizer.encode(F2)] = [2.0, 1.0]
    a.observe(F, 0, -1.0, F2, terminated=True)
    assert a.q[a.discretizer.encode(F), 0] == pytest.approx(-0.1)


def test_epsilon_extremes():
    a = agent()
    a.q[a.discretizer.encode(F)] = [0.0, 5.0]
    assert {a.act(F, 0.0) for _ in range(50)} == {1}
    assert {a.act(F, 1.0) for _ in range(200)} == {0, 1}


def test_greedy_is_deterministic_and_leaves_rng_untouched():
    a = agent()
    state_before = a.rng.bit_generator.state
    assert {a.greedy(F) for _ in range(20)} == {0}  # tie -> action 0
    assert a.rng.bit_generator.state == state_before


def test_exploitation_breaks_ties_randomly():
    a = agent()
    assert {a.act(F, 0.0) for _ in range(200)} == {0, 1}


def test_baselines():
    h = HeuristicPolicy()
    assert h(np.array([0, 20.0, 0, 0])) == 1 and h(np.array([0, -20.0, 0, 0])) == 0
    r = RandomPolicy(flap_prob=0.5, seed=0)
    assert {r(F) for _ in range(100)} == {0, 1}


def test_explore_flap_prob_biases_the_exploration_branch():
    a = QLearningAgent(seed=0, explore_flap_prob=0.1)
    flaps = np.mean([a.act(F, 1.0) for _ in range(4000)])
    assert 0.07 < flaps < 0.13
    assert {QLearningAgent(seed=0, explore_flap_prob=0.0).act(F, 1.0) for _ in range(50)} == {0}
