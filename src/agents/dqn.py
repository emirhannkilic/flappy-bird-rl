"""DQN with a replay buffer and a periodically synchronised target network.

Inputs are the selected columns of the feature vector [dx, dy, vy, y], clipped to the
fixed discretiser ranges and scaled to [-1, 1].
"""
import numpy as np
import torch
from torch import nn

from src.discretizer import FEATURE_RANGES

N_ACTIONS = 2


def normalise(features: np.ndarray, columns: tuple[int, ...]) -> np.ndarray:
    """Clip each selected feature to its fixed range and scale it to [-1, 1]."""
    lo = np.array([FEATURE_RANGES[c][0] for c in columns])
    hi = np.array([FEATURE_RANGES[c][1] for c in columns])
    values = np.asarray(features, dtype=np.float64)[list(columns)]
    return (2.0 * (np.clip(values, lo, hi) - lo) / (hi - lo) - 1.0).astype(np.float32)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), np.float32)
        self.action = np.zeros(capacity, np.int64)
        self.reward = np.zeros(capacity, np.float32)
        self.terminated = np.zeros(capacity, np.float32)
        self.size = 0
        self._next = 0

    def add(self, obs, action, reward, next_obs, terminated) -> None:
        i = self._next
        self.obs[i], self.action[i], self.reward[i] = obs, action, reward
        self.next_obs[i], self.terminated[i] = next_obs, float(terminated)
        self._next = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(self.size, size=batch_size)
        return (
            torch.from_numpy(self.obs[idx]),
            torch.from_numpy(self.action[idx]),
            torch.from_numpy(self.reward[idx]),
            torch.from_numpy(self.next_obs[idx]),
            torch.from_numpy(self.terminated[idx]),
        )


class DQNAgent:
    def __init__(
        self,
        columns: tuple[int, ...] = (0, 1, 2),
        lr: float = 3e-4,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_size: int = 50_000,
        warmup_steps: int = 5_000,
        target_sync_steps: int = 2_000,
        train_every: int = 4,
        grad_clip: float = 10.0,
        explore_flap_prob: float = 0.05,
        seed: int = 0,
    ):
        torch.set_num_threads(1)  # tiny network: avoids thread oversubscription across parallel runs
        torch.manual_seed(seed)  # network initialisation
        self.columns = tuple(columns)
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.target_sync_steps = target_sync_steps
        self.train_every = train_every
        self.grad_clip = grad_clip
        self.explore_flap_prob = explore_flap_prob
        self.rng = np.random.default_rng(seed)  # exploration and replay sampling only

        obs_dim = len(self.columns)
        self.online = nn.Sequential(nn.Linear(obs_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, N_ACTIONS))
        self.target = nn.Sequential(nn.Linear(obs_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, N_ACTIONS))
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size, obs_dim)
        self.steps = 0
        self.loss_ema: float | None = None  # exponential moving average of the Huber loss

    def _q_values(self, features: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            return self.online(torch.from_numpy(normalise(features, self.columns)))

    def act(self, features: np.ndarray, epsilon: float) -> int:
        """Biased epsilon-greedy: with prob. epsilon flap w.p. `explore_flap_prob`, else greedy."""
        if self.rng.random() < epsilon:
            return int(self.rng.random() < self.explore_flap_prob)
        return self.greedy(features)

    def greedy(self, features: np.ndarray) -> int:
        """Deterministic evaluation action (ties -> action 0)."""
        return int(torch.argmax(self._q_values(features)))

    def observe(self, features, action, reward, next_features, terminated: bool) -> None:
        self.buffer.add(
            normalise(features, self.columns), action, reward, normalise(next_features, self.columns), terminated
        )
        self.steps += 1
        if self.steps >= self.warmup_steps and self.steps % self.train_every == 0:
            self._update()
        if self.steps % self.target_sync_steps == 0:
            self.target.load_state_dict(self.online.state_dict())

    def _update(self) -> float:
        obs, action, reward, next_obs, terminated = self.buffer.sample(self.batch_size, self.rng)
        with torch.no_grad():
            # Only `terminated` cuts the bootstrap; a truncated episode still has future value.
            target = reward + self.gamma * (1.0 - terminated) * self.target(next_obs).max(dim=1).values
        q = self.online(obs).gather(1, action.unsqueeze(1)).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.optimizer.step()
        value = loss.item()
        self.loss_ema = value if self.loss_ema is None else 0.99 * self.loss_ema + 0.01 * value
        return value

    def diagnostics(self, recent: int = 1024) -> dict:
        """Training-health metrics; deterministic and RNG-free (uses the latest transitions)."""
        size = self.buffer.size
        n = min(size, recent)
        idx = (self.buffer._next - n + np.arange(n)) % self.buffer.capacity
        with torch.no_grad():
            q = self.online(torch.from_numpy(self.buffer.obs[idx])).abs() if size else torch.zeros(1)
        return {
            "buffer_terminal_rate": float(self.buffer.terminated[:size].mean()) if size else 0.0,
            f"recent_terminal_rate_{recent}": float(self.buffer.terminated[idx].mean()) if size else 0.0,
            "loss_ema": self.loss_ema,
            "abs_q_mean": float(q.mean()),
            "abs_q_max": float(q.max()),
        }

    def save(self, path) -> None:
        torch.save({"columns": self.columns, "online": self.online.state_dict(), "steps": self.steps}, path)

    def load(self, path) -> None:
        """Load policy weights into an agent built with the same `columns`."""
        ckpt = torch.load(path, weights_only=True)
        if tuple(ckpt["columns"]) != self.columns:
            raise ValueError(f"checkpoint columns {ckpt['columns']} != agent columns {self.columns}")
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["online"])
