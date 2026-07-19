# Multi-Agent Football RL

This project implements a simplified 2v2 continuous-physics football environment
and PPO self-play trainer.

## Features

- Four player-controlled discs in a rectangular football field.
- Continuous 2D force actions in each agent's egocentric attacking frame.
- Ball movement emerges only from physics collisions.
- Entity-set observations for self, teammate, opponents, ball, own goal, and opponent goal.
- Egocentric observations are flipped so every policy instance attacks in the same direction.
- Sparse team rewards for scoring/conceding/out-of-bounds with small exploration shaping.
- Shared teammate policy trained with PPO, GAE, entropy regularization, clipped updates, and an opponent checkpoint pool.
- Actor-critic network: entity embedding, transformer encoder, global pooling, actor and critic heads.
- Optional pygame visualization during training.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py --updates 200 --rollout-steps 2048
```

Choose the policy architecture:

```bash
python train.py --model transformer
python train.py --model mlp
```

Enable visualization:

```bash
python train.py --visualize --render-fps 60 --render-every 1
```

While the pygame window is open, press `V` to pause/resume drawing or `Esc` to close the window while training continues.

For a quick smoke test:

```bash
python train.py --updates 1 --rollout-steps 64
```

Checkpoints are written to `checkpoints/` by default.

Run visual inference with the latest checkpoint loaded for both teams:

```bash
python inference_test.py --episodes 5
```

`inference_test.py` reads the model type from newer checkpoints. For older checkpoints, it defaults to `transformer`; override it when needed:

```bash
python inference_test.py --model mlp --checkpoint checkpoints/policy_000010.pt
```

Manually test the environment:

```bash
python manual_test.py
```

Default controls use arrow keys for the selected player. Press `1`-`4` to select a player, `Tab` to toggle all-player controls, `R` to reset, `Space` to pause, and `Esc` to close.
