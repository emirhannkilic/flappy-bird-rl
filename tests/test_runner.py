import numpy as np
import pytest

from src.agents.baselines import HeuristicPolicy
from src.agents.qlearning import QLearningAgent
from src.runner import evaluate, linear_epsilon, train


def test_linear_epsilon():
    assert linear_epsilon(0, 1.0, 0.05, 100) == 1.0
    assert linear_epsilon(50, 1.0, 0.05, 100) == pytest.approx(0.525)
    assert linear_epsilon(100, 1.0, 0.05, 100) == pytest.approx(0.05)
    assert linear_epsilon(10_000, 1.0, 0.05, 100) == pytest.approx(0.05)  # stays at the floor


def test_evaluate_is_deterministic_and_reports_stats():
    a = evaluate(HeuristicPolicy(), episodes=3, score_limit=5, seed=1)
    b = evaluate(HeuristicPolicy(), episodes=3, score_limit=5, seed=1)
    assert a == b
    assert len(a["scores"]) == 3 and 0.0 <= a["capped_rate"] <= 1.0
    assert all(s <= 5 for s in a["scores"])


def test_evaluate_capped_rate_counts_truncation():
    stats = evaluate(HeuristicPolicy(), episodes=3, score_limit=2, seed=0)
    assert stats["capped_rate"] > 0 and max(stats["scores"]) == 2


def run(seed):
    agent = QLearningAgent(seed=seed)
    log = train(agent, total_steps=4000, seed=seed, epsilon_decay_steps=2000,
                eval_every=2000, eval_episodes=3, eval_score_limit=5)
    return agent, log


def test_train_log_structure_and_budget():
    _, log = run(0)
    assert [e["step"] for e in log["evals"]] == [2000, 4000]
    assert log["train_episodes"][-1]["end_step"] <= 4000
    assert sum(ep["length"] for ep in log["train_episodes"]) <= 4000


def test_train_is_reproducible_for_same_seed_and_differs_across_seeds():
    a1, l1 = run(0)
    a2, l2 = run(0)
    a3, _ = run(1)
    assert l1 == l2 and np.array_equal(a1.q, a2.q)
    assert not np.array_equal(a1.q, a3.q)


def test_checkpoints_saved_at_requested_steps_and_diagnostics_logged(tmp_path):
    from src.agents.dqn import DQNAgent

    agent = DQNAgent(warmup_steps=200, seed=0)
    log = train(agent, total_steps=600, seed=0, epsilon_decay_steps=300, eval_every=300, eval_episodes=1,
                eval_score_limit=2, checkpoint_steps=(300, 600), checkpoint_dir=tmp_path, checkpoint_prefix="run")
    assert sorted(f.name for f in tmp_path.iterdir()) == ["run_step300.ckpt", "run_step600.ckpt"]
    assert "diagnostics" in log["evals"][0] and "abs_q_max" in log["evals"][0]["diagnostics"]
    with pytest.raises(ValueError):
        train(QLearningAgent(), 10, 0, 5, checkpoint_steps=(5,))


def test_qlearning_checkpoint_keeps_exact_filename_and_reloads(tmp_path):
    agent = QLearningAgent(seed=0)
    train(agent, total_steps=300, seed=0, epsilon_decay_steps=100, eval_every=300, eval_episodes=1,
          eval_score_limit=2, checkpoint_steps=(300,), checkpoint_dir=tmp_path, checkpoint_prefix="q")
    assert [f.name for f in tmp_path.iterdir()] == ["q_step300.ckpt"]  # no '.npy' suffix appended
    fresh = QLearningAgent(seed=1)
    fresh.load(tmp_path / "q_step300.ckpt")
    assert np.array_equal(fresh.q, agent.q)
