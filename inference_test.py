from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import torch

from football_rl import Football2v2Env, FootballConfig, EntityTransformerActorCritic
from football_rl.render import PygameFootballRenderer, RenderConfig


CHECKPOINT_RE = re.compile(r"policy_(\d+)\.pt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run visual inference with the latest checkpoint on both teams.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--render-fps", type=int, default=60)
    parser.add_argument("--render-every", type=int, default=1)
    return parser.parse_args()


def latest_checkpoint(checkpoint_dir: Path) -> Path:
    candidates: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob("policy_*.pt"):
        match = CHECKPOINT_RE.match(path.name)
        if match is not None:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No policy_*.pt checkpoints found in {checkpoint_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def load_policy(path: Path, env: Football2v2Env, device: str) -> EntityTransformerActorCritic:
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model = EntityTransformerActorCritic(num_entity_types=env.num_entity_types).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def run_episode(
    env: Football2v2Env,
    model: EntityTransformerActorCritic,
    renderer: PygameFootballRenderer,
    device: str,
) -> dict[str, object]:
    obs = env.reset()
    done = False
    info: dict[str, object] = {"event": None}
    team_returns = np.zeros(2, dtype=np.float32)

    while not done and not renderer.closed:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        actions, _, _ = model.act(obs_tensor, deterministic=False)
        obs, rewards, done, info = env.step(actions.cpu().numpy())
        team_returns[0] += float(rewards[:2].mean())
        team_returns[1] += float(rewards[2:].mean())
        info = dict(info)
        info["team0_return"] = float(team_returns[0])
        info["team1_return"] = float(team_returns[1])
        renderer.render(info)

    info = dict(info)
    info["team0_return"] = float(team_returns[0])
    info["team1_return"] = float(team_returns[1])
    return info


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint or latest_checkpoint(args.checkpoint_dir)
    env = Football2v2Env(FootballConfig(max_steps=args.max_steps), seed=args.seed)
    model = load_policy(checkpoint_path, env, args.device)
    renderer = PygameFootballRenderer(
        env,
        RenderConfig(fps=args.render_fps, render_every=args.render_every, enabled=True),
    )

    print(f"Loaded checkpoint: {checkpoint_path}")
    try:
        for episode in range(1, args.episodes + 1):
            info = run_episode(env, model, renderer, args.device)
            print(
                f"episode={episode} "
                f"event={info.get('event')} "
                f"team0_return={info['team0_return']:.3f} "
                f"team1_return={info['team1_return']:.3f}"
            )
            if renderer.closed:
                break
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
