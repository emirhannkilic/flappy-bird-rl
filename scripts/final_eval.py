"""Final evaluation of saved checkpoints and baselines.

python -m scripts.final_eval --root results/main --seeds 0 1 2 3 4

Every (method, seed, step) is evaluated greedily on the same protocol: 100 episodes,
score_limit=500, eval env seed = run seed + EVAL_SEED_OFFSET. The statistical unit is the
training seed: first the mean over the 100 episodes of one seed, then mean +- sample std
(ddof=1) over seeds. Episodes are never pooled across seeds.
"""
import argparse
import json
import warnings
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from src.agents.baselines import HeuristicPolicy, RandomPolicy
from src.agents.dqn import DQNAgent
from src.agents.qlearning import QLearningAgent
from src.discretizer import BINS_CORE, Discretizer
from src.runner import EVAL_SEED_OFFSET, evaluate

EPISODES = 100
SCORE_LIMIT = 500
CHECKPOINT_STEPS = (500_000, 1_000_000, 2_000_000, 3_000_000)
BASELINES = {"random_uniform": 0.5, "random_biased": 0.1, "heuristic": None}


def load_policy(method: str, root: Path, seed: int, step: int):
    path = root / method / "checkpoints" / f"{method}_seed{seed}_step{step}.ckpt"
    agent = DQNAgent(seed=seed) if method == "dqn" else QLearningAgent(Discretizer(BINS_CORE), seed=seed)
    agent.load(path)
    return agent.greedy


def run_job(job):
    warnings.filterwarnings("ignore")
    kind, name, seed, step, root = job
    if kind == "checkpoint":
        policy = load_policy(name, Path(root), seed, step)
    else:
        policy = HeuristicPolicy() if name == "heuristic" else RandomPolicy(BASELINES[name], seed=seed)
    stats = evaluate(policy, EPISODES, SCORE_LIMIT, seed + EVAL_SEED_OFFSET)
    return {"method": name, "seed": seed, "step": step, **{k: stats[k] for k in ("mean_score", "std_score", "capped_rate", "mean_length")}}


def summarise(results):
    rows = {}
    for r in results:
        rows.setdefault((r["method"], r["step"]), []).append(r)
    out = []
    for (method, step), rs in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        score = np.array([r["mean_score"] for r in rs])
        capped = np.array([r["capped_rate"] for r in rs])
        out.append({
            "method": method, "step": step, "n_seeds": len(rs),
            "score_mean": float(score.mean()), "score_std": float(score.std(ddof=1)) if len(rs) > 1 else 0.0,
            "capped_mean": float(capped.mean()), "capped_std": float(capped.std(ddof=1)) if len(rs) > 1 else 0.0,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results/main"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, nargs="+", default=list(CHECKPOINT_STEPS))
    ap.add_argument("--methods", nargs="+", default=["qlearning", "dqn"])
    args = ap.parse_args()

    jobs = [("checkpoint", m, s, t, str(args.root)) for m, s, t in product(args.methods, args.seeds, args.steps)]
    jobs += [("baseline", b, s, None, str(args.root)) for b, s in product(BASELINES, args.seeds)]
    with Pool(8) as pool:
        results = pool.map(run_job, jobs, chunksize=1)
    summary = summarise(results)
    (args.root / "final_eval.json").write_text(json.dumps({"per_seed": results, "summary": summary}, indent=1))
    for row in summary:
        step = f"{row['step']:>9}" if row["step"] else "        -"
        print(f"{row['method']:15s} {step}  score {row['score_mean']:6.1f} +- {row['score_std']:5.1f}  "
              f"capped {row['capped_mean']:.2f} +- {row['capped_std']:.2f}  (n={row['n_seeds']})")


if __name__ == "__main__":
    main()
