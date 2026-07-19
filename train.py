from __future__ import annotations

import argparse

from football_rl import Football2v2Env, FootballConfig, PPOConfig, SelfPlayPPOTrainer, build_actor_critic
from football_rl.render import PygameFootballRenderer, RenderConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 2v2 football self-play PPO.")
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument("--render-fps", type=int, default=60)
    parser.add_argument("--model", choices=("transformer", "mlp"), default="transformer")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = Football2v2Env(FootballConfig(), seed=args.seed)
    model = build_actor_critic(args.model, num_entity_types=env.num_entity_types, num_entities=env.num_entities)
    print(model)
    print("Parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    cfg = PPOConfig(
        total_updates=args.updates,
        rollout_steps=args.rollout_steps,
        device=args.device,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
    )
    renderer = None
    if args.visualize:
        renderer = PygameFootballRenderer(
            env,
            RenderConfig(fps=args.render_fps, render_every=args.render_every, enabled=True),
        )
    trainer = SelfPlayPPOTrainer(env, model, cfg, renderer=renderer)
    trainer.train()


if __name__ == "__main__":
    main()
