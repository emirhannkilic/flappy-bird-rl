"""CLI: python -m src.train --agent {qlearning,dqn} --seed 0 --steps 3000000"""
import argparse
import json
from pathlib import Path

from src.agents.qlearning import QLearningAgent
from src.agents.dqn import DQNAgent
from src.discretizer import (
    BINS_COARSE,
    BINS_CORE,
    BINS_CORE_COARSE,
    BINS_CORE_FINE,
    BINS_DEFAULT,
    BINS_FINE,
    BINS_Y3,
    Discretizer,
)
from src.runner import train

BIN_PRESETS = {
    "core": BINS_CORE,  # selected by tuning (720 states)
    "core_coarse": BINS_CORE_COARSE,
    "core_fine": BINS_CORE_FINE,
    "y3": BINS_Y3,
    "y6": BINS_COARSE,
    "legacy36k": BINS_DEFAULT,
    "legacy150k": BINS_FINE,
}


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
    if args.agent == "dqn":
        return DQNAgent(
            columns=(0, 1, 2, 3) if args.use_y else (0, 1, 2),
            lr=args.lr,
            gamma=args.gamma,
            explore_flap_prob=args.explore_flap_prob,
            seed=args.seed,
        )
    raise ValueError(f"unknown agent: {args.agent}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--agent", choices=["qlearning", "dqn"], default="qlearning")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=3_000_000)
    p.add_argument("--eps-decay-steps", type=int, default=300_000)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--explore-flap-prob", type=float, default=0.05)
    p.add_argument("--tie-break", choices=["random", "idle"], default="idle")
    p.add_argument("--bins", choices=sorted(BIN_PRESETS), default="core")
    p.add_argument("--lr", type=float, default=3e-4, help="DQN only")
    p.add_argument("--use-y", action="store_true", help="DQN only: add the absolute height y as a 4th input")
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
