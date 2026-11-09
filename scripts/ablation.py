"""Ablation runner (protocol locked in CLAUDE.md, 'Ablation protokolü').

python -m scripts.ablation --configs ql --seeds 0 1 2 3 4                       # full Q-learning ablations
python -m scripts.ablation --configs all --seeds 0 --steps 60000 --root results/ablation_smoke  # smoke test

Layout: <root>/<config>/{config.json, final_eval.json, <method>/{<method>_seed<N>.json, log_<N>.txt, checkpoints/}}.
Runs whose output JSON already exists are skipped, so an interrupted batch can be restarted.
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# name -> (method, extra `src.train` args, extra `scripts.final_eval` args); everything else is the baseline.
CONFIGS = {
    "ql_res210": ("qlearning", ["--bins", "core_coarse"], ["--bins", "core_coarse"]),
    "ql_res3600": ("qlearning", ["--bins", "core_fine"], ["--bins", "core_fine"]),
    "ql_y3": ("qlearning", ["--bins", "y3"], ["--bins", "y3"]),
    "ql_y6": ("qlearning", ["--bins", "y6"], ["--bins", "y6"]),
    "ql_noalive": ("qlearning", ["--no-alive-reward"], []),
    "dqn_usey": ("dqn", ["--use-y"], ["--use-y"]),
    "dqn_noalive": ("dqn", ["--no-alive-reward"], []),
}
for _prefix, _method in (("ql", "qlearning"), ("dqn", "dqn")):
    for _p in ("0.02", "0.10", "0.50"):
        CONFIGS[f"{_prefix}_pflap{_p}"] = (_method, ["--explore-flap-prob", _p], [])

EVAL_EPISODES = 50  # periodic eval, same as the main experiment


def git_state() -> dict:
    run = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    return {"git_head": run("rev-parse", "HEAD"), "git_dirty": bool(run("status", "--porcelain"))}


def checkpoint_steps(steps: int) -> list[int]:
    return [steps // 6, steps // 3, 2 * steps // 3, steps]  # 3M -> 500k / 1M / 2M / 3M


def train_cmd(method, extra, seed, steps, cfg_dir):
    mdir = cfg_dir / method
    return [
        sys.executable, "-m", "src.train", "--agent", method, "--seed", str(seed), "--steps", str(steps),
        "--eval-every", str(steps // 12), "--eval-episodes", str(EVAL_EPISODES),
        "--checkpoint-steps", *map(str, checkpoint_steps(steps)), "--checkpoint-dir", str(mdir / "checkpoints"),
        "--out", str(mdir / f"{method}_seed{seed}.json"), *extra,
    ]


def run_train(cmd, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log:
        return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["all"], help="config names, or ql / dqn / all")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--root", type=Path, default=Path("results/ablation"))
    ap.add_argument("--procs", type=int, default=8)
    args = ap.parse_args()
    assert args.steps % 12 == 0, "steps must be divisible by 12 (eval every steps/12, checkpoints at steps/6 multiples)"

    names = []
    for c in args.configs:
        names += [n for n in CONFIGS if c in ("all", n) or (c == "ql" and CONFIGS[n][0] == "qlearning") or (c == "dqn" and CONFIGS[n][0] == "dqn")]
    names = list(dict.fromkeys(names))
    unknown = [c for c in args.configs if c not in ("all", "ql", "dqn") and c not in CONFIGS]
    if unknown:
        sys.exit(f"unknown configs: {unknown}; choose from {sorted(CONFIGS)}")

    jobs = []
    for name in names:
        method, extra, _ = CONFIGS[name]
        cfg_dir = args.root / name
        cfg_dir.mkdir(parents=True, exist_ok=True)
        (cfg_dir / "config.json").write_text(json.dumps({
            "config": name, "method": method, "train_extra_args": extra, "steps": args.steps,
            "seeds": args.seeds, "checkpoint_steps": checkpoint_steps(args.steps), **git_state(),
        }, indent=1))
        for seed in args.seeds:
            cmd = train_cmd(method, extra, seed, args.steps, cfg_dir)
            if not (cfg_dir / method / f"{method}_seed{seed}.json").exists():
                jobs.append((name, seed, cmd, cfg_dir / method / f"log_{seed}.txt"))
    print(f"{len(names)} configs, {len(jobs)} runs to train ({args.procs} parallel)", flush=True)

    with ThreadPoolExecutor(args.procs) as pool:
        codes = list(pool.map(lambda j: run_train(j[2], j[3]), jobs))
    failed = [(j[0], j[1]) for j, c in zip(jobs, codes) if c != 0]
    if failed:
        sys.exit(f"training failed (see log_*.txt): {failed}")

    for name in names:
        method, _, eval_extra = CONFIGS[name]
        cmd = [sys.executable, "-m", "scripts.final_eval", "--root", str(args.root / name), "--methods", method,
               "--seeds", *map(str, args.seeds), "--steps", *map(str, checkpoint_steps(args.steps)),
               "--no-baselines", *eval_extra]
        print(f"final_eval {name}", flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
