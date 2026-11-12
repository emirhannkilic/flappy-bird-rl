"""Learning-curve figure and step-to-threshold table from the main experiment.

python -m scripts.report --root results/main --out results/report

Curve data: periodic evals in <root>/<method>/<method>_seed<N>.json (greedy, 50 episodes,
score_limit=100). Statistical unit is the training seed: mean +- sample std (ddof=1) over seeds.
Step-to-threshold = first periodic eval whose mean score reaches THRESHOLD. It is a first-hit
measure (a later collapse is not counted against it) with EVAL_EVERY resolution; seeds that never
reach the threshold are censored and reported as a count, never averaged in as 3M.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = {"qlearning": "Q-learning", "dqn": "DQN"}
THRESHOLD = 50  # above the hand-written heuristic (31 at cap 500), reachable under the cap-100 periodic eval


def load_runs(root: Path, method: str) -> list[dict]:
    return [json.load(open(p)) for p in sorted((root / method).glob(f"{method}_seed*.json"))]


def step_to_threshold(evals: list[dict], threshold: float = THRESHOLD) -> int | None:
    """First eval step with mean_score >= threshold, or None if never reached."""
    return next((e["step"] for e in evals if e["mean_score"] >= threshold), None)


def curve(runs: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps = [e["step"] for e in runs[0]["evals"]]
    assert all([e["step"] for e in r["evals"]] == steps for r in runs), "eval steps differ across seeds"
    scores = np.array([[e["mean_score"] for e in r["evals"]] for r in runs])
    return np.array(steps), scores.mean(0), scores.std(0, ddof=1)


TABLE_ROWS = [  # (method, step, label); steps are the pre-registered 1M and 3M checkpoints
    ("random_uniform", None, "Random (uniform)"), ("random_biased", None, "Random (biased, $p=0.1$)"),
    ("heuristic", None, "Heuristic"), ("qlearning", 1_000_000, "Q-learning, 1M"),
    ("qlearning", 3_000_000, "Q-learning, 3M"), ("dqn", 1_000_000, "DQN, 1M"), ("dqn", 3_000_000, "DQN, 3M"),
]


def main_table(per_seed: list[dict]) -> str:
    """LaTeX rows: score, capped rate (%), episode length; mean +- sample std over seeds (final_eval, cap 500)."""
    def cell(vals, scale=1.0, fmt=".1f"):
        v = np.asarray(vals) * scale
        return f"${v.mean():{fmt}} \\pm {v.std(ddof=1):{fmt}}$"

    lines = []
    for method, step, label in TABLE_ROWS:
        rows = [r for r in per_seed if r["method"] == method and r["step"] == step]
        assert len(rows) >= 2, f"no per-seed rows for {method} @ {step}"
        lines.append(" & ".join([label, cell([r["mean_score"] for r in rows]),
                                 cell([r["capped_rate"] for r in rows], 100), cell([r["mean_length"] for r in rows], fmt=".0f")]) + r" \\")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results/main"))
    ap.add_argument("--out", type=Path, default=Path("results/report"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    rows = []
    for method, label in METHODS.items():
        runs = load_runs(args.root, method)
        steps, mean, std = curve(runs)
        ax.plot(steps / 1e6, mean, marker="o", ms=3, label=label)
        ax.fill_between(steps / 1e6, mean - std, mean + std, alpha=0.2)
        hits = [step_to_threshold(r["evals"]) for r in runs]
        reached = [h for h in hits if h is not None]
        rows.append({
            "method": method, "n_seeds": len(runs), "threshold": THRESHOLD, "reached": len(reached),
            "step_mean": float(np.mean(reached)) if reached else None,
            "step_std": float(np.std(reached, ddof=1)) if len(reached) > 1 else None,
            "per_seed": hits,
        })
    ax.axhline(THRESHOLD, color="gray", ls=":", lw=1, label=f"threshold ({THRESHOLD})")
    ax.set_xlabel("Training steps (millions)")
    ax.set_ylabel("Mean score (greedy, 50 ep, cap 100)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "learning_curves.pdf")
    fig.savefig(args.out / "learning_curves.png", dpi=200)
    json.dump(rows, open(args.out / "step_to_threshold.json", "w"), indent=1)
    final = json.load(open(args.root / "final_eval.json"))
    (args.out / "main_table.tex").write_text(main_table(final["per_seed"]) + "\n")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
