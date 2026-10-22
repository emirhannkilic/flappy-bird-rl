"""CLI: python -m src.train --agent qlearning --seed 0 --steps 1000000"""
import argparse
import json
from pathlib import Path

from src.agents.qlearning import QLearningAgent
from src.discretizer import BINS_COARSE, BINS_DEFAULT, BINS_FINE, Discretizer
from src.runner import train

BIN_PRESETS = {"coarse": BINS_COARSE, "default": BINS_DEFAULT, "fine": BINS_FINE}


def build_agent(args):
    if args.agent == "qlearning":
        return QLearningAgent(
            Discretizer(BIN_PRESETS[args.bins]),
            alpha=args.alpha,
            gamma=args.gamma,
            seed=args.seed,
            explore_flap_prob=args.explore_flap_prob,
            tie_break=args.tie_break,
        )
    raise ValueError(f"unknown agent: {args.agent}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--agent", choices=["qlearning"], default="qlearning")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--eps-decay-steps", type=int, default=300_000)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--explore-flap-prob", type=float, default=0.5)
    p.add_argument("--tie-break", choices=["random", "idle"], default="random")
    p.add_argument("--bins", choices=sorted(BIN_PRESETS), default="default")
    p.add_argument("--eval-every", type=int, default=50_000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    log = train(
        build_agent(args),
        total_steps=args.steps,
        seed=args.seed,
        epsilon_decay_steps=args.eps_decay_steps,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
    )
    log["config"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    out = args.out or Path("results") / f"{args.agent}_{args.bins}_seed{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(log))
    last = log["evals"][-1]
    print(f"{out}: final eval mean score {last['mean_score']:.1f}, capped {last['capped_rate']:.2f}")


if __name__ == "__main__":
    main()
