"""Q-learning tuning on tuning seeds (100-102), disjoint from the reported seeds.

python -m scripts.tune tiebreak            # isolate exploitation tie-break vs. table size
python -m scripts.tune grid --tie-break idle   # bins x explore_flap_prob, then top-2 at 3M steps
"""
import argparse
import json
import warnings
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from src.agents.qlearning import QLearningAgent
from src.discretizer import BINS_COARSE, BINS_DEFAULT, Discretizer
from src.runner import train

TUNING_SEEDS = (100, 101, 102)
EVAL_EPISODES = 50
EPS_DECAY_STEPS = 300_000
BINS_720, BINS_2160 = (8, 9, 10, 1), (8, 9, 10, 3)
GRID_BINS = (BINS_720, BINS_2160, BINS_COARSE, BINS_DEFAULT)  # 720 / 2160 / 4320 / 36000
GRID_FLAP_PROBS = (0.02, 0.05, 0.10)
OUT_DIR = Path("results/tuning")


def run_one(job):
    bins, flap_prob, tie_break, seed, steps = job
    warnings.filterwarnings("ignore")
    agent = QLearningAgent(
        Discretizer(bins), seed=seed, explore_flap_prob=flap_prob, tie_break=tie_break
    )
    log = train(agent, steps, seed, EPS_DECAY_STEPS, eval_every=1_000_000, eval_episodes=EVAL_EPISODES)
    return {
        "bins": bins, "flap_prob": flap_prob, "tie_break": tie_break, "seed": seed, "steps": steps,
        "evals": [{k: e[k] for k in ("step", "mean_score", "capped_rate", "mean_length")} for e in log["evals"]],
    }


def summarise(results, keys):
    groups = {}
    for r in results:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    rows = []
    for key, rs in groups.items():
        final = [r["evals"][-1]["mean_score"] for r in rs]
        rows.append((key, float(np.mean(final)), float(np.std(final)), final))
    return sorted(rows, key=lambda row: -row[1])


def print_rows(rows, keys):
    for key, mean, std, per_seed in rows:
        label = " ".join(f"{k}={v}" for k, v in zip(keys, key))
        print(f"{label:55s} mean {mean:6.1f} +- {std:5.1f}  seeds {[round(x, 1) for x in per_seed]}", flush=True)


def pmap(jobs):
    with Pool(8) as pool:
        return pool.map(run_one, jobs, chunksize=1)


def cmd_tiebreak(_):
    jobs = [(b, 0.05, tb, s, 1_000_000) for b, tb, s in product((BINS_2160, BINS_DEFAULT), ("random", "idle"), TUNING_SEEDS)]
    results = pmap(jobs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "tiebreak.json").write_text(json.dumps(results))
    print_rows(summarise(results, ("bins", "tie_break")), ("bins", "tie_break"))


def cmd_grid(args):
    keys = ("bins", "flap_prob")
    stage1 = pmap([(b, p, args.tie_break, s, 1_000_000) for b, p, s in product(GRID_BINS, GRID_FLAP_PROBS, TUNING_SEEDS)])
    rows1 = summarise(stage1, keys)
    print("== stage 1: 1M steps ==")
    print_rows(rows1, keys)
    top2 = [row[0] for row in rows1[:2]]
    stage2 = pmap([(b, p, args.tie_break, s, 3_000_000) for (b, p), s in product(top2, TUNING_SEEDS)])
    print("== stage 2: top-2 at 3M steps (eval at 1M/2M/3M) ==")
    print_rows(summarise(stage2, keys), keys)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"grid_{args.tie_break}.json").write_text(json.dumps({"stage1": stage1, "stage2": stage2}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tiebreak").set_defaults(fn=cmd_tiebreak)
    g = sub.add_parser("grid")
    g.add_argument("--tie-break", choices=["random", "idle"], required=True)
    g.set_defaults(fn=cmd_grid)
    args = ap.parse_args()
    args.fn(args)
