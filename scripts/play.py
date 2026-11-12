"""Failure analysis and videos for trained agents.

python -m scripts.play --method dqn --step 3000000 --seeds 0 1 2 3 4 --video death --video-dir docs/videos

Rollouts follow the final-eval protocol (100 episodes, score_limit=500, eval seed = run seed +
EVAL_SEED_OFFSET), so episode scores match `final_eval.json`. For every episode that ends in a
crash, the cause (ground / upper pipe / lower pipe) and the features at the crash are recorded.
Capped episodes did not fail and are counted separately. Videos replay the same deterministic
rollout with rendering switched on for one episode only (frames are drawn on explicit
`env.render()` calls, so rendering never changes the trajectory).
"""
import argparse
import json
import subprocess
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pygame
from flappy_bird_gymnasium.envs.constants import PLAYER_HEIGHT
from PIL import Image

from scripts.final_eval import CHECKPOINT_STEPS, EPISODES, SCORE_LIMIT, load_policy
from src.env import make_env
from src.features import PLAYER_RIGHT, SCREEN_HEIGHT, extract_features, select_target_pipe
from src.runner import EVAL_SEED_OFFSET


def _load_png_with_pillow(path, namehint=""):
    """The pygame 2.6.1 wheel on Python 3.14 ships without SDL_image (BMP only), but the env's
    sprites are PNG. Pillow (already installed with matplotlib) decodes them instead."""
    img = Image.open(path).convert("RGBA")
    return pygame.image.frombytes(img.tobytes(), img.size, "RGBA")


if not pygame.image.get_extended():
    pygame.image.load = _load_png_with_pillow

GROUND_Y = SCREEN_HEIGHT * 0.79  # same formula as the environment (404.48)
FPS = 30
FRAME_SIZE = (288, 512)


def classify_death(obs: np.ndarray) -> str:
    """Crash cause from the observation returned by the terminating step.

    The observation already reflects the crash state, so the colliding pipe is the target pipe
    (first real pipe whose right edge is still ahead of the bird's left edge).
    """
    y = float(obs[9])
    if y + PLAYER_HEIGHT >= GROUND_Y - 1:  # same test as the environment
        return "ground"
    pipe_x, gap_top, gap_bottom = select_target_pipe(obs)
    if pipe_x < PLAYER_RIGHT:  # horizontally overlapping the target pipe
        if y < gap_top:
            return "pipe_top"
        if y + PLAYER_HEIGHT > gap_bottom:
            return "pipe_bottom"
    return "other"


def rollout(policy, episodes: int, seed: int, score_limit: int | None = SCORE_LIMIT, render: tuple | None = None):
    """Greedy rollout. `render=(episode, first_step, last_step)` also returns the frames of that
    episode between those steps (inclusive); no frames are drawn otherwise."""
    env = make_env(score_limit, render_mode="rgb_array" if render else None)
    obs, _ = env.reset(seed=seed)
    records, frames = [], []
    for episode in range(episodes):
        if episode > 0:
            obs, _ = env.reset()
        step = 0
        while True:
            if render and episode == render[0] and render[1] <= step <= render[2]:
                frames.append(env.render())
            obs, _, terminated, truncated, info = env.step(policy(extract_features(obs)))
            step += 1
            if terminated or truncated:
                break
        if render and episode == render[0]:
            frames.append(env.render())  # final frame: the crash itself
        features = extract_features(obs)
        records.append({
            "episode": episode, "score": info["score"], "length": step, "capped": bool(truncated),
            "cause": None if truncated else classify_death(obs),
            "dx": float(features[0]), "dy": float(features[1]), "vy": float(features[2]),
        })
    env.close()
    return records, frames


def pick_episode(records: list[dict], kind: str) -> dict | None:
    """First crashed episode ('death') or first capped episode ('success'); None if absent."""
    want_capped = kind == "success"
    return next((r for r in records if r["capped"] == want_capped), None)


def frame_window(length: int, max_frames: int, kind: str) -> tuple[int, int]:
    """Steps to render: the start of a success, the last `max_frames` steps of a death."""
    return (0, max_frames - 1) if kind == "success" else (max(0, length - max_frames), length)


def write_video(frames: list[np.ndarray], path: Path) -> None:
    """H.264 mp4 via ffmpeg (no Python dependency); yuv420p keeps it playable in browsers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{FRAME_SIZE[0]}x{FRAME_SIZE[1]}", "-r", str(FPS), "-i", "-",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    subprocess.run(cmd, input=b"".join(np.ascontiguousarray(f, dtype=np.uint8).tobytes() for f in frames), check=True)


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results/main"))
    ap.add_argument("--method", choices=["qlearning", "dqn"], required=True)
    ap.add_argument("--step", type=int, choices=CHECKPOINT_STEPS, default=CHECKPOINT_STEPS[-1])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--episodes", type=int, default=EPISODES)
    ap.add_argument("--out", type=Path, default=Path("results/report"))
    ap.add_argument("--video", choices=["death", "success"], help="record one episode per seed")
    ap.add_argument("--episode", type=int, help="record this eval episode instead of the first death/success "
                    "(same index + same seed = same start across checkpoints)")
    ap.add_argument("--video-dir", type=Path, default=Path("docs/videos"))
    ap.add_argument("--max-frames", type=int, default=900, help="video length cap in steps (30 s)")
    args = ap.parse_args()

    runs = {}
    for seed in args.seeds:
        policy = load_policy(args.method, args.root, seed, args.step)
        records, _ = rollout(policy, args.episodes, seed + EVAL_SEED_OFFSET)
        runs[seed] = records
        if args.video:
            chosen = records[args.episode] if args.episode is not None else pick_episode(records, args.video)
            if chosen is None:
                print(f"seed {seed}: no {args.video} episode to record")
                continue
            outcome = "success" if chosen["capped"] else "death"  # decides window and file name
            window = frame_window(chosen["length"], args.max_frames, outcome)
            _, frames = rollout(policy, chosen["episode"] + 1, seed + EVAL_SEED_OFFSET, render=(chosen["episode"], *window))
            name = f"{args.method}_seed{seed}_step{args.step}_ep{chosen['episode']}_{outcome}.mp4"
            write_video(frames, args.video_dir / name)
            print(f"seed {seed}: episode {chosen['episode']} ({outcome}, score {chosen['score']}, length {chosen['length']}) -> {args.video_dir / name}")

    # Statistical unit is the seed: causes are counted per crashed episode, then summed for display.
    summary = {
        "method": args.method, "step": args.step, "episodes_per_seed": args.episodes,
        "per_seed": {str(s): {"capped": sum(r["capped"] for r in rs), "causes": dict(Counter(r["cause"] for r in rs if not r["capped"]))}
                     for s, rs in runs.items()},
        "episodes": {str(s): rs for s, rs in runs.items()},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"deaths_{args.method}_step{args.step}.json").write_text(json.dumps(summary, indent=1))
    for s, v in summary["per_seed"].items():
        print(f"seed {s}: capped {v['capped']:3d}  crashes {v['causes']}")


if __name__ == "__main__":
    main()
