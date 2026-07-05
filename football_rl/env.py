from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class EntityType(IntEnum):
    SELF = 0
    TEAMMATE = 1
    OPPONENT = 2
    BALL = 3
    OWN_GOAL = 4
    OPPONENT_GOAL = 5


@dataclass
class FootballConfig:
    field_length: float = 24.0
    field_width: float = 14.0
    goal_width: float = 5.0
    player_radius: float = 0.45
    ball_radius: float = 0.28
    player_mass: float = 1.0
    ball_mass: float = 0.35
    player_friction: float = 0.02
    ball_friction: float = 0.7
    max_force: float = 24.0
    max_player_speed: float = 7.0
    max_ball_speed: float = 11.0
    dt: float = 0.05
    physics_substeps: int = 4
    max_steps: int = 600
    score_reward: float = 10.0
    concede_reward: float = -10.0
    out_reward: float = -1.0
    opponent_out_reward: float = 1.0
    ball_approach_weight: float = 0.015
    ball_progress_weight: float = 0.15
    action_penalty_weight: float = 0.001


class Football2v2Env:
    """A lightweight continuous-physics 2v2 football environment.

    There are four player-controlled discs and one ball disc. Actions are 2D
    forces. Ball movement only comes from rigid-body collision impulses.
    """

    num_teams = 2
    players_per_team = 2
    num_agents = 4
    num_entities = 7
    numeric_features = 4
    num_entity_types = len(EntityType)

    def __init__(self, config: FootballConfig | None = None, seed: int | None = None):
        self.cfg = config or FootballConfig()
        self.rng = np.random.default_rng(seed)
        self.player_pos = np.zeros((4, 2), dtype=np.float32)
        self.player_vel = np.zeros((4, 2), dtype=np.float32)
        self.ball_pos = np.zeros(2, dtype=np.float32)
        self.ball_vel = np.zeros(2, dtype=np.float32)
        self.prev_ball_x = 0.0
        self.steps = 0
        self.last_info: dict[str, object] = {}

    @property
    def observation_space_shape(self) -> tuple[int, int]:
        return (self.num_entities, self.numeric_features + 1)

    @property
    def action_space_shape(self) -> tuple[int]:
        return (2,)

    def reset(self) -> np.ndarray:
        x = self.cfg.field_length * 0.25
        y = self.cfg.field_width * 0.18
        base = np.array([[-x, -y], [-x, y], [x, -y], [x, y]], dtype=np.float32)
        noise = self.rng.normal(0.0, 0.7, size=(4, 2)).astype(np.float32)
        self.player_pos = base + noise
        self.player_vel.fill(0.0)
        self.ball_pos = self.rng.normal(0.0, 2, size=2).astype(np.float32)
        self.ball_vel.fill(0.0)
        self.prev_ball_x = float(self.ball_pos[0])
        self.steps = 0
        self.last_info = {}
        return self.observe()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, dict[str, object]]:
        actions = np.asarray(actions, dtype=np.float32).reshape(4, 2)
        actions = np.clip(actions, -1.0, 1.0) * self.cfg.max_force
        world_actions = actions.copy()
        world_actions[2:, 0] *= -1.0
        prev_dist = self._agent_ball_distances()
        self.prev_ball_x = float(self.ball_pos[0])

        sub_dt = self.cfg.dt / self.cfg.physics_substeps
        for _ in range(self.cfg.physics_substeps):
            self._integrate(world_actions, sub_dt)
            self._resolve_collisions()

        self.steps += 1
        event, scoring_team, offending_team = self._terminal_event()
        done = event is not None or self.steps >= self.cfg.max_steps
        rewards = self._rewards(actions, prev_dist, event, scoring_team, offending_team)
        info = {
            "event": event or ("timeout" if done else None),
            "scoring_team": scoring_team,
            "offending_team": offending_team,
            "steps": self.steps,
        }
        self.last_info = info
        return self.observe(), rewards.astype(np.float32), done, info

    def observe(self) -> np.ndarray:
        return np.stack([self._observe_agent(i) for i in range(4)]).astype(np.float32)

    def team_of(self, agent_idx: int) -> int:
        return 0 if agent_idx < 2 else 1

    def _observe_agent(self, agent_idx: int) -> np.ndarray:
        team = self.team_of(agent_idx)
        side = 1.0 if team == 0 else -1.0
        self_pos = self.player_pos[agent_idx]
        self_vel = self.player_vel[agent_idx]
        teammate = 1 - agent_idx if team == 0 else 5 - agent_idx
        opponents = [2, 3] if team == 0 else [0, 1]
        own_goal = np.array([-self.cfg.field_length / 2, 0.0], dtype=np.float32)
        opp_goal = np.array([self.cfg.field_length / 2, 0.0], dtype=np.float32)
        if team == 1:
            own_goal, opp_goal = opp_goal, own_goal

        entities = [
            self._entity(self_pos, self_vel, self_pos, self_vel, side, EntityType.SELF),
            self._entity(self.player_pos[teammate], self.player_vel[teammate], self_pos, self_vel, side, EntityType.TEAMMATE),
            self._entity(self.player_pos[opponents[0]], self.player_vel[opponents[0]], self_pos, self_vel, side, EntityType.OPPONENT),
            self._entity(self.player_pos[opponents[1]], self.player_vel[opponents[1]], self_pos, self_vel, side, EntityType.OPPONENT),
            self._entity(self.ball_pos, self.ball_vel, self_pos, self_vel, side, EntityType.BALL),
            self._entity(own_goal, np.zeros(2, dtype=np.float32), self_pos, self_vel, side, EntityType.OWN_GOAL),
            self._entity(opp_goal, np.zeros(2, dtype=np.float32), self_pos, self_vel, side, EntityType.OPPONENT_GOAL),
        ]
        return np.stack(entities)

    def _entity(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        origin_pos: np.ndarray,
        origin_vel: np.ndarray,
        side: float,
        entity_type: EntityType,
    ) -> np.ndarray:
        rel_pos = (pos - origin_pos).astype(np.float32)
        rel_vel = (vel - origin_vel).astype(np.float32)
        rel_pos[0] *= side
        rel_vel[0] *= side
        scale = np.array(
            [
                self.cfg.field_length,
                self.cfg.field_width,
                self.cfg.max_ball_speed,
                self.cfg.max_ball_speed,
                1.0,
            ],
            dtype=np.float32,
        )
        return np.array([rel_pos[0], rel_pos[1], rel_vel[0], rel_vel[1], float(entity_type)], dtype=np.float32) / scale

    def _integrate(self, actions: np.ndarray, dt: float) -> None:
        self.player_vel += (actions / self.cfg.player_mass) * dt
        self.player_vel *= self.cfg.player_friction ** dt
        self.ball_vel *= self.cfg.ball_friction ** dt
        self.player_vel = self._clip_norm(self.player_vel, self.cfg.max_player_speed)
        self.ball_vel = self._clip_norm(self.ball_vel[None, :], self.cfg.max_ball_speed)[0]
        self.player_pos += self.player_vel * dt
        self.ball_pos += self.ball_vel * dt

    def _resolve_collisions(self) -> None:
        for i in range(4):
            for j in range(i + 1, 4):
                self._resolve_pair(i, j, self.cfg.player_radius, self.cfg.player_radius)
            self._resolve_ball_player(i)

    def _resolve_pair(self, i: int, j: int, ri: float, rj: float) -> None:
        pi = self.player_pos[i]
        pj = self.player_pos[j]
        vi = self.player_vel[i]
        vj = self.player_vel[j]
        self._apply_disc_impulse(pi, pj, vi, vj, ri, rj, self.cfg.player_mass, self.cfg.player_mass)

    def _resolve_ball_player(self, i: int) -> None:
        self._apply_disc_impulse(
            self.player_pos[i],
            self.ball_pos,
            self.player_vel[i],
            self.ball_vel,
            self.cfg.player_radius,
            self.cfg.ball_radius,
            self.cfg.player_mass,
            self.cfg.ball_mass,
        )

    def _apply_disc_impulse(
        self,
        pos_a: np.ndarray,
        pos_b: np.ndarray,
        vel_a: np.ndarray,
        vel_b: np.ndarray,
        radius_a: float,
        radius_b: float,
        mass_a: float,
        mass_b: float,
    ) -> None:
        delta = pos_b - pos_a
        dist = float(np.linalg.norm(delta))
        min_dist = radius_a + radius_b
        if dist >= min_dist:
            return
        if dist < 1e-6:
            normal = self.rng.normal(size=2).astype(np.float32)
            normal /= np.linalg.norm(normal) + 1e-6
        else:
            normal = delta / dist

        penetration = min_dist - max(dist, 1e-6)
        total_mass = mass_a + mass_b
        pos_a -= normal * penetration * (mass_b / total_mass)
        pos_b += normal * penetration * (mass_a / total_mass)

        rel_vel = vel_b - vel_a
        normal_speed = float(np.dot(rel_vel, normal))
        if normal_speed > 0.0:
            return
        restitution = 0.75
        impulse = -(1.0 + restitution) * normal_speed / (1.0 / mass_a + 1.0 / mass_b)
        vel_a -= (impulse / mass_a) * normal
        vel_b += (impulse / mass_b) * normal

    def _terminal_event(self) -> tuple[str | None, int | None, int | None]:
        half_l = self.cfg.field_length / 2
        half_w = self.cfg.field_width / 2
        if abs(float(self.ball_pos[0])) > half_l:
            if abs(float(self.ball_pos[1])) <= self.cfg.goal_width / 2:
                return "goal", 0 if self.ball_pos[0] > 0 else 1, None
            return "ball_out", None, None
        if abs(float(self.ball_pos[1])) > half_w:
            return "ball_out", None, None
        out_x = np.abs(self.player_pos[:, 0]) > half_l
        out_y = np.abs(self.player_pos[:, 1]) > half_w
        out = np.flatnonzero(out_x | out_y)
        if out.size:
            return "player_out", None, self.team_of(int(out[0]))
        return None, None, None

    def _rewards(
        self,
        actions: np.ndarray,
        prev_dist: np.ndarray,
        event: str | None,
        scoring_team: int | None,
        offending_team: int | None,
    ) -> np.ndarray:
        rewards = np.zeros(4, dtype=np.float32)
        if event == "goal" and scoring_team is not None:
            rewards[:2] += self.cfg.score_reward if scoring_team == 0 else self.cfg.concede_reward
            rewards[2:] += self.cfg.score_reward if scoring_team == 1 else self.cfg.concede_reward
        elif event == "ball_out":
            rewards -= self.cfg.out_reward * -1.0
        elif event == "player_out" and offending_team is not None:
            rewards[:2] += self.cfg.out_reward if offending_team == 0 else self.cfg.opponent_out_reward
            rewards[2:] += self.cfg.out_reward if offending_team == 1 else self.cfg.opponent_out_reward

        new_dist = self._agent_ball_distances()
        rewards += self.cfg.ball_approach_weight * (prev_dist - new_dist)
        ball_dx = float(self.ball_pos[0] - self.prev_ball_x)
        rewards[:2] += self.cfg.ball_progress_weight * ball_dx
        rewards[2:] -= self.cfg.ball_progress_weight * ball_dx
        rewards -= self.cfg.action_penalty_weight * np.linalg.norm(actions / self.cfg.max_force, axis=1)
        return rewards

    def _agent_ball_distances(self) -> np.ndarray:
        return np.linalg.norm(self.player_pos - self.ball_pos[None, :], axis=1).astype(np.float32)

    @staticmethod
    def _clip_norm(values: np.ndarray, max_norm: float) -> np.ndarray:
        norms = np.linalg.norm(values, axis=-1, keepdims=True)
        scale = np.minimum(1.0, max_norm / (norms + 1e-8))
        return values * scale
