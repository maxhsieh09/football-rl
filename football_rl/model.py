from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal


class SquashedGaussianActorCritic(nn.Module):
    model_type = "base"

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean, std, _ = self(obs)
        return Normal(mean, std)

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std, value = self(obs)
        dist = Normal(mean, std)
        raw_action = mean if deterministic else dist.rsample()
        action = torch.tanh(raw_action)
        log_prob = self._squashed_log_prob(dist, raw_action, action)
        return action, log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std, value = self(obs)
        dist = Normal(mean, std)
        clipped = actions.clamp(-0.999, 0.999)
        raw_actions = torch.atanh(clipped)
        log_prob = self._squashed_log_prob(dist, raw_actions, clipped)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value

    @staticmethod
    def _squashed_log_prob(dist: Normal, raw_action: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        correction = torch.log(1.0 - action.pow(2) + 1e-6)
        return (dist.log_prob(raw_action) - correction).sum(dim=-1)


class SinusoidalNumericEmbedding(nn.Module):
    def __init__(
        self,
        numeric_dim: int,
        hidden_dim: int,
        sinusoidal_dim: int = 32,
        max_period: float = 10_000.0,
        value_scale: float = 100.0,
    ):
        super().__init__()
        if sinusoidal_dim % 2 != 0:
            raise ValueError("sinusoidal_dim must be even")
        self.numeric_dim = numeric_dim
        self.sinusoidal_dim = sinusoidal_dim
        self.value_scale = value_scale

        half_dim = sinusoidal_dim // 2
        freqs = torch.exp(-torch.log(torch.tensor(max_period)) * torch.arange(half_dim) / half_dim)
        self.register_buffer("freqs", freqs, persistent=False)
        self.mlp = nn.Sequential(
            nn.Linear(numeric_dim * sinusoidal_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        angles = values.unsqueeze(-1) * self.value_scale * self.freqs
        sinusoidal = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.mlp(sinusoidal.flatten(start_dim=-2))


class EntityTransformerActorCritic(SquashedGaussianActorCritic):
    model_type = "transformer"

    def __init__(
        self,
        num_entity_types: int,
        numeric_dim: int = 4,
        action_dim: int = 2,
        hidden_dim: int = 16,
        num_layers: int = 2,
        num_heads: int = 4,
        numeric_sinusoidal_dim: int = 16,
    ):
        super().__init__()
        self.type_embedding = nn.Embedding(num_entity_types, hidden_dim)
        self.numeric_embedding = SinusoidalNumericEmbedding(
            numeric_dim=numeric_dim,
            hidden_dim=hidden_dim,
            sinusoidal_dim=numeric_sinusoidal_dim,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.actor = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), 0.))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        numeric = obs[..., :4]
        type_ids = obs[..., 4].round().long().clamp_min(0)
        x = self.numeric_embedding(numeric) + self.type_embedding(type_ids)
        x = self.encoder(x)
        pooled = x.mean(dim=1)
        mean = self.actor(pooled)
        value = self.critic(pooled).squeeze(-1)
        std = self.log_std.exp().expand_as(mean)
        return mean, std, value


class MLPActorCritic(SquashedGaussianActorCritic):
    model_type = "mlp"

    def __init__(
        self,
        num_entities: int = 7,
        entity_dim: int = 5,
        action_dim: int = 2,
        hidden_dim: int = 32,
        num_layers: int = 2,
    ):
        super().__init__()
        input_dim = num_entities * entity_dim
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
        for _ in range(max(0, num_layers - 1)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()])
        self.backbone = nn.Sequential(*layers)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), 0.))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = obs.flatten(start_dim=1)
        x = self.backbone(x)
        mean = self.actor(x)
        value = self.critic(x).squeeze(-1)
        std = self.log_std.exp().expand_as(mean)
        return mean, std, value


def build_actor_critic(model_type: str, num_entity_types: int, num_entities: int = 7) -> SquashedGaussianActorCritic:
    if model_type == "transformer":
        return EntityTransformerActorCritic(num_entity_types=num_entity_types)
    if model_type == "mlp":
        return MLPActorCritic(num_entities=num_entities)
    raise ValueError(f"Unknown model type: {model_type}")
