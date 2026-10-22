"""Method-agnostic training and evaluation loop.

Every agent implements:
    act(features, epsilon) -> int         exploratory action
    greedy(features) -> int               deterministic evaluation action
    observe(features, action, reward, next_features, terminated) -> None
so all methods share the same budget, exploration schedule and evaluation protocol.
"""
import random

import numpy as np
import torch

from src.env import make_env
from src.features import extract_features

EVAL_SEED_OFFSET = 10_000  # evaluation env seed = run seed + offset (never the training seed)


def linear_epsilon(step: int, start: float, end: float, decay_steps: int) -> float:
    frac = min(step / decay_steps, 1.0)
    return start + frac * (end - start)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(policy, episodes: int, score_limit: int | None, seed: int) -> dict:
    """Run `policy(features) -> action` greedily; the env is re-seeded once at the start,
    so the result is a deterministic function of the policy."""
    env = make_env(score_limit)
    obs, _ = env.reset(seed=seed)
    scores, lengths, capped = [], [], 0
    for episode in range(episodes):
        if episode > 0:
            obs, _ = env.reset()
        length = 0
        while True:
            obs, _, terminated, truncated, info = env.step(policy(extract_features(obs)))
            length += 1
            if terminated or truncated:
                break
        scores.append(info["score"])
        lengths.append(length)
        capped += int(truncated)
    env.close()
    return {
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "capped_rate": capped / episodes,
        "mean_length": float(np.mean(lengths)),
        "scores": scores,
    }


def train(
    agent,
    total_steps: int,
    seed: int,
    epsilon_decay_steps: int,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    eval_every: int = 50_000,
    eval_episodes: int = 20,
    train_score_limit: int | None = 100,
    eval_score_limit: int | None = 100,
) -> dict:
    """Train for exactly `total_steps` environment steps; evaluate every `eval_every` steps
    and once at the end. Returns a JSON-serialisable log."""
    seed_everything(seed)
    env = make_env(train_score_limit)
    obs, _ = env.reset(seed=seed)  # seeded once per run, never per episode
    features = extract_features(obs)

    train_episodes, evals = [], []
    ep_return, ep_length = 0.0, 0
    for step in range(total_steps):
        epsilon = linear_epsilon(step, epsilon_start, epsilon_end, epsilon_decay_steps)
        action = agent.act(features, epsilon)
        obs, reward, terminated, truncated, info = env.step(action)
        next_features = extract_features(obs)
        agent.observe(features, action, reward, next_features, terminated)
        ep_return += reward
        ep_length += 1
        features = next_features

        if terminated or truncated:
            train_episodes.append(
                {"end_step": step + 1, "score": info["score"], "length": ep_length, "return": ep_return}
            )
            ep_return, ep_length = 0.0, 0
            obs, _ = env.reset()
            features = extract_features(obs)

        if (step + 1) % eval_every == 0 or step + 1 == total_steps:
            stats = evaluate(agent.greedy, eval_episodes, eval_score_limit, seed + EVAL_SEED_OFFSET)
            evals.append({"step": step + 1, **stats})
    env.close()
    return {"seed": seed, "total_steps": total_steps, "train_episodes": train_episodes, "evals": evals}
