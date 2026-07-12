from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import copy
import random

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from football_rl.env import Football2v2Env
from football_rl.model import EntityTransformerActorCritic
from football_rl.render import PygameFootballRenderer


@dataclass
class PPOConfig:
    rollout_steps: int = 2048
    total_updates: int = 200
    gamma: float = 0.995
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 512
    checkpoint_interval: int = 4
    opponent_pool_size: int = 16
    device: str = "cpu"
    seed: int = 1
    checkpoint_dir: str = "checkpoints"


@dataclass
class RolloutBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    values: torch.Tensor


class SelfPlayPPOTrainer:
    def __init__(
        self,
        env: Football2v2Env,
        model: EntityTransformerActorCritic,
        config: PPOConfig,
        renderer: PygameFootballRenderer | None = None,
    ):
        self.env = env
        self.model = model.to(config.device)
        self.cfg = config
        self.renderer = renderer
        self.optimizer = Adam(self.model.parameters(), lr=config.learning_rate)
        self.rng = random.Random(config.seed)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        self.opponent_pool: deque[dict[str, torch.Tensor]] = deque(maxlen=config.opponent_pool_size)
        self.opponent = copy.deepcopy(self.model).to(config.device).eval()
        self._fill_initial_opponent_pool()
        self.sample_opponent()
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> None:
        obs = self.env.reset()
        try:
            for update in range(1, self.cfg.total_updates + 1):
                batch, obs, stats = self.collect_rollout(obs)
                metrics = self.update(batch)
                if update % self.cfg.checkpoint_interval == 0:
                    self.save_checkpoint(update)
                print(
                    f"update={update} "
                    f"return={stats['episode_return']:.3f} "
                    f"goals={stats['goals']} "
                    f"policy_loss={metrics['policy_loss']:.4f} "
                    f"value_loss={metrics['value_loss']:.4f} "
                    f"entropy={metrics['entropy']:.4f}"
                )
        finally:
            if self.renderer is not None:
                self.renderer.close()

    @torch.no_grad()
    def collect_rollout(self, obs: np.ndarray) -> tuple[RolloutBatch, np.ndarray, dict[str, float]]:
        obs_buf: list[np.ndarray] = []
        action_buf: list[np.ndarray] = []
        log_prob_buf: list[np.ndarray] = []
        value_buf: list[np.ndarray] = []
        reward_buf: list[np.ndarray] = []
        done_buf: list[bool] = []
        episode_return = 0.0
        team_episode_returns = np.zeros(2, dtype=np.float32)
        goals = 0

        for _ in range(self.cfg.rollout_steps):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.cfg.device)
            train_obs = obs_tensor[:2]
            opp_obs = obs_tensor[2:]
            train_action, train_log_prob, train_value = self.model.act(train_obs)
            opp_action, _, _ = self.opponent.act(opp_obs)
            actions = torch.cat([train_action, opp_action], dim=0).cpu().numpy()

            next_obs, rewards, done, info = self.env.step(actions)
            team_episode_returns[0] += float(rewards[:2].mean())
            team_episode_returns[1] += float(rewards[2:].mean())
            info = dict(info)
            info["team0_return"] = float(team_episode_returns[0])
            info["team1_return"] = float(team_episode_returns[1])
            if self.renderer is not None:
                self.renderer.render(info)
            obs_buf.append(obs[:2].copy())
            action_buf.append(actions[:2].copy())
            log_prob_buf.append(train_log_prob.cpu().numpy())
            value_buf.append(train_value.cpu().numpy())
            reward_buf.append(rewards[:2].copy())
            done_buf.append(done)
            episode_return += float(rewards[:2].mean())
            goals += int(info.get("event") == "goal")

            if done:
                self.sample_opponent()
                obs = self.env.reset()
                team_episode_returns.fill(0.0)
            else:
                obs = next_obs

        obs_tensor = torch.as_tensor(np.asarray(obs_buf), dtype=torch.float32, device=self.cfg.device).reshape(-1, self.env.num_entities, 5)
        action_tensor = torch.as_tensor(np.asarray(action_buf), dtype=torch.float32, device=self.cfg.device).reshape(-1, 2)
        old_log_probs = torch.as_tensor(np.asarray(log_prob_buf), dtype=torch.float32, device=self.cfg.device).reshape(-1)
        values = torch.as_tensor(np.asarray(value_buf), dtype=torch.float32, device=self.cfg.device)
        rewards = torch.as_tensor(np.asarray(reward_buf), dtype=torch.float32, device=self.cfg.device)
        dones = torch.as_tensor(np.asarray(done_buf), dtype=torch.float32, device=self.cfg.device)

        next_value = torch.zeros(2, dtype=torch.float32, device=self.cfg.device)
        if not bool(done_buf[-1]):
            next_obs = torch.as_tensor(obs[:2], dtype=torch.float32, device=self.cfg.device)
            _, _, next_value = self.model.act(next_obs, deterministic=True)

        returns, advantages = self.compute_gae(rewards, values, dones, next_value)
        batch = RolloutBatch(
            obs=obs_tensor,
            actions=action_tensor,
            log_probs=old_log_probs,
            returns=returns.reshape(-1),
            advantages=advantages.reshape(-1),
            values=values.reshape(-1),
        )
        stats = {"episode_return": episode_return / max(1, self.cfg.rollout_steps), "goals": float(goals)}
        return batch, obs, stats

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(rewards)
        last_advantage = torch.zeros(2, dtype=torch.float32, device=self.cfg.device)
        for t in reversed(range(rewards.shape[0])):
            next_nonterminal = 1.0 - dones[t]
            bootstrap = next_value if t == rewards.shape[0] - 1 else values[t + 1]
            delta = rewards[t] + self.cfg.gamma * bootstrap * next_nonterminal - values[t]
            last_advantage = delta + self.cfg.gamma * self.cfg.gae_lambda * next_nonterminal * last_advantage
            advantages[t] = last_advantage
        returns = advantages + values
        flat = advantages.reshape(-1)
        advantages = (advantages - flat.mean()) / (flat.std(unbiased=False) + 1e-8)
        return returns, advantages

    def update(self, batch: RolloutBatch) -> dict[str, float]:
        n = batch.obs.shape[0]
        indices = torch.arange(n, device=self.cfg.device)
        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        updates = 0
        for _ in range(self.cfg.update_epochs):
            perm = indices[torch.randperm(n, device=self.cfg.device)]
            for start in range(0, n, self.cfg.minibatch_size):
                idx = perm[start : start + self.cfg.minibatch_size]
                log_probs, entropy, values = self.model.evaluate_actions(batch.obs[idx], batch.actions[idx])
                ratio = (log_probs - batch.log_probs[idx]).exp()
                unclipped = ratio * batch.advantages[idx]
                clipped = torch.clamp(ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio) * batch.advantages[idx]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = nn.functional.mse_loss(values, batch.returns[idx])
                entropy_loss = entropy.mean()
                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                metrics["policy_loss"] += float(policy_loss.detach().cpu())
                metrics["value_loss"] += float(value_loss.detach().cpu())
                metrics["entropy"] += float(entropy_loss.detach().cpu())
                updates += 1
        return {k: v / max(1, updates) for k, v in metrics.items()}

    def save_checkpoint(self, update: int) -> None:
        state = self._cpu_state_dict()
        self.opponent_pool.append(state)
        torch.save({"update": update, "model": state}, self.checkpoint_dir / f"policy_{update:06d}.pt")

    def sample_opponent(self) -> None:
        if not self.opponent_pool:
            self.opponent.load_state_dict(self.model.state_dict())
            return
        state = self.rng.choice(list(self.opponent_pool))
        self.opponent.load_state_dict(state)
        self.opponent.eval()

    def _cpu_state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def _fill_initial_opponent_pool(self) -> None:
        for idx in range(self.cfg.opponent_pool_size):
            self.opponent_pool.append(self._random_opponent_state_dict(self.cfg.seed + 10_000 + idx))

    def _random_opponent_state_dict(self, seed: int) -> dict[str, torch.Tensor]:
        opponent = copy.deepcopy(self.model).cpu()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            opponent.apply(self._reset_module_parameters)
        return {k: v.detach().cpu().clone() for k, v in opponent.state_dict().items()}

    @staticmethod
    def _reset_module_parameters(module: nn.Module) -> None:
        reset_parameters = getattr(module, "reset_parameters", None)
        if callable(reset_parameters):
            reset_parameters()
