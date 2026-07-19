"""Multi-agent self-play football RL components."""

from football_rl.env import Football2v2Env, FootballConfig
from football_rl.model import EntityTransformerActorCritic, MLPActorCritic, build_actor_critic
from football_rl.ppo import PPOConfig, SelfPlayPPOTrainer

__all__ = [
    "Football2v2Env",
    "FootballConfig",
    "EntityTransformerActorCritic",
    "MLPActorCritic",
    "build_actor_critic",
    "PPOConfig",
    "SelfPlayPPOTrainer",
]
