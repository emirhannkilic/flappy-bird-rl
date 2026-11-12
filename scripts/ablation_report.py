"""Ablation table (LaTeX) and figure from the 3M final evaluations.

python -m scripts.ablation_report --main results/main --ablation results/ablation --out results/report

Every ablation config is compared with the reference run of the same method (`results/main`:
720 states, no y, alive reward on, p_flap 0.05). Statistical unit is the training seed (n=5):
per-seed score = mean over 100 final-eval episodes (cap 500). The Welch p-values are two-sided and
NOT corrected for the 12 comparisons; with n=5 they are a rough guide, not a verdict.
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STEP = 3_000_000
REFERENCE = {"qlearning": "Reference (720 states)", "dqn": "Reference (3 features)"}
CONFIGS = {  # method -> [(config dir, label)], in table order
    "qlearning": [
        ("ql_res210", "210 states (coarser)"), ("ql_res3600", "3600 states (finer)"),
        ("ql_y3", "+ y, 2160 states"), ("ql_y6", "+ y, 4320 states"), ("ql_noalive", "No alive reward"),
        ("ql_pflap0.02", r"$p_{\mathrm{flap}}=0.02$"), ("ql_pflap0.10", r"$p_{\mathrm{flap}}=0.10$"),
        ("ql_pflap0.50", r"$p_{\mathrm{flap}}=0.50$"),
    ],
    "dqn": [
        ("dqn_usey", "+ y input"), ("dqn_noalive", "No alive reward"),
        ("dqn_pflap0.02", r"$p_{\mathrm{flap}}=0.02$"), ("dqn_pflap0.10", r"$p_{\mathrm{flap}}=0.10$"),
        ("dqn_pflap0.50", r"$p_{\mathrm{flap}}=0.50$"),
    ],
}
METHOD_NAMES = {"qlearning": "Q-learning", "dqn": "DQN"}


def seed_scores(path: Path, method: str, step: int = STEP) -> list[float]:
    """Per-seed mean score at `step`, ordered by seed."""
    rows = [r for r in json.load(open(path))["per_seed"] if r["method"] == method and r["step"] == step]
    return [r["mean_score"] for r in sorted(rows, key=lambda r: r["seed"])]


def welch(a, b) -> tuple[float, float, float]:
    """Welch's t-test: (t, degrees of freedom, two-sided p). Stdlib + numpy only (no scipy installed)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    t = (a.mean() - b.mean()) / math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    # p = 1 - P(|T| < |t|); integrate the Student-t density over the finite range [0, |t|].
    xs = np.linspace(0.0, abs(t), 20001)
    log_c = math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - 0.5 * math.log(df * math.pi)
    pdf = np.exp(log_c - (df + 1) / 2 * np.log1p(xs**2 / df))
    return t, df, float(1.0 - 2.0 * np.trapezoid(pdf, xs))


def collect(main: Path, ablation: Path) -> dict[str, list[tuple[str, str, list[float]]]]:
    """method -> [(config, label, per-seed scores)], reference first."""
    out = {}
    for method, configs in CONFIGS.items():
        rows = [("reference", REFERENCE[method], seed_scores(main / "final_eval.json", method))]
        rows += [(c, label, seed_scores(ablation / c / "final_eval.json", method)) for c, label in configs]
        for name, _, scores in rows:
            assert len(scores) == 5, f"{name}: expected 5 seeds, got {len(scores)}"
        out[method] = rows
    return out


def latex_table(data) -> str:
    lines = []
    for method, rows in data.items():
        ref = np.asarray(rows[0][2])
        lines.append(rf"\multicolumn{{4}}{{l}}{{\textit{{{METHOD_NAMES[method]}}}}} \\")
        for name, label, scores in rows:
            s = np.asarray(scores)
            if name == "reference":
                lines.append(rf"{label} & ${s.mean():.1f} \pm {s.std(ddof=1):.1f}$ & -- & -- \\")
                continue
            t, df, p = welch(s, ref)
            lines.append(rf"{label} & ${s.mean():.1f} \pm {s.std(ddof=1):.1f}$ & ${s.mean() - ref.mean():+.1f}$ & ${p:.2f}$ \\")
    return "\n".join(lines)


def figure(data, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, gridspec_kw={"width_ratios": [8, 5]})
    for ax, (method, rows) in zip(axes, data.items()):
        ref_mean = np.mean(rows[0][2])
        for i, (name, label, scores) in enumerate(rows[::-1]):
            y = np.full(len(scores), i)
            ax.scatter(scores, y, s=14, color="tab:gray" if name == "reference" else "tab:blue", alpha=0.7, zorder=3)
            ax.plot([np.mean(scores)] * 2, [i - 0.3, i + 0.3], color="black", lw=2, zorder=4)
        ax.axvline(ref_mean, color="tab:red", ls=":", lw=1)
        ax.set_yticks(range(len(rows)), [r[1] for r in rows[::-1]], fontsize=8)
        ax.set_title(METHOD_NAMES[method], fontsize=10)
        ax.set_xlabel("Score at 3M (per-seed mean)")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=200)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", type=Path, default=Path("results/main"))
    ap.add_argument("--ablation", type=Path, default=Path("results/ablation"))
    ap.add_argument("--out", type=Path, default=Path("results/report"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = collect(args.main, args.ablation)
    table = latex_table(data)
    (args.out / "ablation_table.tex").write_text(table + "\n")
    figure(data, args.out / "ablation")
    print(table)


if __name__ == "__main__":
    main()
