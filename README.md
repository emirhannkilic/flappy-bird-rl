# Flappy Bird RL

Tabular Q-learning and DQN on `flappy-bird-gymnasium` (`FlappyBird-v0`), compared on the same three state features, the same exploration schedule and the same evaluation protocol. Course project for CMP4501.

Both agents see three features derived from the raw observation: the horizontal distance to the pipe that currently constrains the bird (`dx`), the vertical offset to the gap centre (`dy`) and the vertical velocity (`vy`). Q-learning discretises them into 720 states; DQN takes them as scaled inputs to a 3-64-64-2 network. The absolute height `y` is only used in ablations.

## Setup

Python 3.14 (versions in `requirements.txt` are pinned and were verified on macOS).

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

Videos need `ffmpeg` on the path. The pygame 2.6.1 wheel on Python 3.14 cannot load PNG files, so `scripts/play.py` decodes the sprites with Pillow (installed with matplotlib).

## Running

Everything writes to `results/`, which is not tracked. One run of the main experiment (seed 0, 3M steps) looks like this:

```
python -m src.train --agent qlearning --seed 0 --steps 3000000 \
  --eval-every 250000 --eval-episodes 50 \
  --checkpoint-steps 500000 1000000 2000000 3000000 \
  --checkpoint-dir results/main/qlearning/checkpoints \
  --out results/main/qlearning/qlearning_seed0.json
```

Use `--agent dqn` and a `dqn` path for the DQN run. The main experiment uses seeds 0 to 4. Hyperparameter tuning used seeds 100 to 102 so tuning never touches the reported seeds.

| Command | Purpose |
|---|---|
| `python -m scripts.final_eval --root results/main` | Greedy evaluation of every checkpoint and the baselines: 100 episodes, score limit 500, eval seed = run seed + 10000 |
| `python -m scripts.ablation --configs all` | The 13 ablation configs, 5 seeds each, including their final evaluation (resumable) |
| `python -m scripts.report` | Learning curves, step-to-threshold, main results table |
| `python -m scripts.ablation_report` | Ablation table and figure |
| `python -m scripts.play --method dqn --step 3000000 --seeds 2` | Crash analysis; add `--video success --episode 0` to render an episode |
| `python -m scripts.tune` | Q-learning tuning grid |

## Results

Mean score (pipes passed) over 5 seeds, plus or minus the standard deviation across seeds. Each seed's score is the mean of 100 greedy episodes with a limit of 500 pipes.

| Agent | Score at 1M steps | Score at 3M steps |
|---|---|---|
| Random (uniform) | 0.0 | 0.0 |
| Heuristic (flap if `dy > 8`) | 31.4 ± 2.2 | 31.4 ± 2.2 |
| Q-learning | 74.7 ± 39.3 | 84.7 ± 61.1 |
| DQN | 81.2 ± 42.1 | 253.5 ± 192.3 |

DQN has the higher mean at 3M, but the spread across seeds is large (one DQN seed scores 57, another 472) and the two methods' learning curves overlap at every evaluation point, so we do not claim one is better. DQN scores also move a lot between checkpoints of the same run. No ablation survives a correction for multiple comparisons; the clearest observations are that a finer Q-table (3600 states) and adding `y` to the DQN input both lowered scores (Welch p of 0.05 and 0.06, uncorrected).

## Layout

```
src/        env wrapper, features, discretizer, agents, training loop, CLI
scripts/    tuning, ablation, final evaluation, report and video scripts
tests/      pytest suite
docs/       GitHub Pages site and demo videos
```

Video file names encode method, seed, checkpoint and episode, for example `dqn_seed2_step3000000_ep0_success.mp4`. Success videos show the first 900 steps of an episode that reaches the 500 pipe limit, and death videos show the last 900 steps before the crash.
