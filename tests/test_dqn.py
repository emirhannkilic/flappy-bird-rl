import numpy as np
import pytest
import torch

from src.agents.dqn import DQNAgent, ReplayBuffer, normalise
from src.discretizer import FEATURE_RANGES

F = np.array([10.0, -5.0, 3.0, 150.0])


def small_agent(**kw):
    kw = {"warmup_steps": 8, "batch_size": 8, "train_every": 1, "target_sync_steps": 4, "seed": 0, **kw}
    return DQNAgent(**kw)


def test_normalise_scales_to_unit_interval_and_clips():
    lows = np.array([lo for lo, _ in FEATURE_RANGES])
    highs = np.array([hi for _, hi in FEATURE_RANGES])
    assert np.allclose(normalise(lows, (0, 1, 2, 3)), -1.0)
    assert np.allclose(normalise(highs, (0, 1, 2, 3)), 1.0)
    assert np.allclose(normalise(lows - 1e6, (0, 1, 2)), -1.0)
    assert np.allclose(normalise(highs + 1e6, (0, 1, 2)), 1.0)
    out = normalise(F, (0, 1, 2))
    assert out.shape == (3,) and out.dtype == np.float32


def test_network_input_matches_feature_columns():
    assert small_agent().online[0].in_features == 3
    assert small_agent(columns=(0, 1, 2, 3)).online[0].in_features == 4


def test_replay_buffer_wraps_around():
    buf = ReplayBuffer(3, 2)
    for i in range(5):
        buf.add(np.full(2, i, np.float32), i % 2, float(i), np.full(2, i + 1, np.float32), i == 4)
    assert buf.size == 3
    assert sorted(buf.reward.tolist()) == [2.0, 3.0, 4.0]
    obs, action, reward, next_obs, term = buf.sample(4, np.random.default_rng(0))
    assert obs.shape == (4, 2) and term.dtype == torch.float32


def test_no_update_before_warmup_then_weights_change():
    a = small_agent()
    before = [p.clone() for p in a.online.parameters()]
    for _ in range(7):
        a.observe(F, 1, 0.1, F, False)
    assert all(torch.equal(b, p) for b, p in zip(before, a.online.parameters()))
    for _ in range(5):
        a.observe(F, 1, 0.1, F, False)
    assert not all(torch.equal(b, p) for b, p in zip(before, a.online.parameters()))


def test_target_network_syncs_on_schedule():
    a = small_agent(target_sync_steps=4, warmup_steps=1000)  # no learning, so only the sync matters
    with torch.no_grad():
        for p in a.online.parameters():
            p.add_(1.0)
    assert not torch.equal(a.online[0].weight, a.target[0].weight)
    for _ in range(4):
        a.observe(F, 0, 0.0, F, False)
    assert torch.equal(a.online[0].weight, a.target[0].weight)


def test_truncated_bootstraps_terminated_does_not():
    """Online net outputs 0, target net outputs 10 everywhere, reward 1, gamma 0.5.

    Bootstrapped target = 1 + 0.5 * 10 = 6 -> Huber loss 5.5; terminal target = 1 -> loss 0.5.
    Runs the agent's real `_update`.
    """
    def update_loss(terminated):
        a = small_agent(gamma=0.5)
        for p in list(a.online.parameters()) + list(a.target.parameters()):
            torch.nn.init.zeros_(p)
        with torch.no_grad():
            a.target[-1].bias.fill_(10.0)
        a.buffer.add(normalise(F, a.columns), 0, 1.0, normalise(F, a.columns), terminated)
        return a._update()

    assert update_loss(terminated=False) == pytest.approx(5.5)  # truncation keeps the bootstrap
    assert update_loss(terminated=True) == pytest.approx(0.5)  # termination cuts it


def test_biased_exploration_and_greedy_determinism():
    a = small_agent(explore_flap_prob=0.1)
    flaps = np.mean([a.act(F, 1.0) for _ in range(4000)])
    assert 0.07 < flaps < 0.13
    state = a.rng.bit_generator.state
    assert len({a.greedy(F) for _ in range(20)}) == 1
    assert a.rng.bit_generator.state == state


def test_same_seed_gives_identical_weights():
    def run(seed):
        a = small_agent(seed=seed)
        for i in range(40):
            a.observe(F + i, i % 2, 0.1, F + i + 1, i % 17 == 0)
        return torch.cat([p.flatten() for p in a.online.parameters()])

    assert torch.equal(run(0), run(0))
    assert not torch.equal(run(0), run(1))


def test_diagnostics_reports_terminal_fraction_and_q_stats():
    a = small_agent(buffer_size=100)
    assert a.diagnostics()["buffer_terminal_rate"] == 0.0  # empty buffer is safe
    for i in range(40):
        a.observe(F + i, i % 2, 0.1, F + i + 1, i % 4 == 0)
    d = a.diagnostics()
    assert d["buffer_terminal_rate"] == pytest.approx(10 / 40)
    assert a.diagnostics(recent=8)["recent_terminal_rate_8"] == pytest.approx(2 / 8)  # last 8 of 40: i=32..39 -> 32, 36
    assert d["loss_ema"] is not None and d["loss_ema"] >= 0
    assert 0 <= d["abs_q_mean"] <= d["abs_q_max"]


def test_diagnostics_does_not_touch_the_rng():
    a = small_agent()
    for i in range(20):
        a.observe(F + i, 0, 0.1, F + i, False)
    state = a.rng.bit_generator.state
    a.diagnostics()
    assert a.rng.bit_generator.state == state


def test_save_load_roundtrip_and_column_mismatch(tmp_path):
    a = small_agent(seed=0)
    for i in range(20):
        a.observe(F + i, i % 2, 0.1, F + i + 1, False)
    a.save(tmp_path / "a.ckpt")
    b = small_agent(seed=5)
    b.load(tmp_path / "a.ckpt")
    assert all(torch.equal(x, y) for x, y in zip(a.online.parameters(), b.online.parameters()))
    assert a.greedy(F) == b.greedy(F)
    with pytest.raises(ValueError):
        small_agent(columns=(0, 1, 2, 3)).load(tmp_path / "a.ckpt")
