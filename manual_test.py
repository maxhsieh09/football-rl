from __future__ import annotations

import argparse
import time

import numpy as np

from football_rl import Football2v2Env, FootballConfig
from football_rl.render import PygameFootballRenderer, RenderConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manually test the 2v2 football environment.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--force", type=float, default=1.0)
    parser.add_argument("--control-all", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


class ManualController:
    def __init__(self, env: Football2v2Env, renderer: PygameFootballRenderer, force: float, control_all: bool):
        self.env = env
        self.renderer = renderer
        self.force = force
        self.control_all = control_all
        self.selected_agent = 0
        self.paused = False
        self.team_returns = np.zeros(2, dtype=np.float32)
        self.last_event: str | None = None

    def run(self) -> None:
        self.env.reset()
        try:
            while not self.renderer.closed:
                action = self._read_input()
                if not self.paused:
                    _, rewards, done, info = self.env.step(action)
                    self.team_returns[0] += float(rewards[:2].mean())
                    self.team_returns[1] += float(rewards[2:].mean())
                    self.last_event = str(info.get("event")) if info.get("event") is not None else None
                    if done:
                        self._render(info)
                        time.sleep(0.6)
                        self.reset()
                        continue
                self._render({"event": self.last_event})
        finally:
            self.renderer.close()

    def reset(self) -> None:
        self.env.reset()
        self.team_returns.fill(0.0)
        self.last_event = None

    def _read_input(self) -> np.ndarray:
        pg = self.renderer.pygame
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.renderer.close()
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self.renderer.close()
                elif event.key == pg.K_r:
                    self.reset()
                elif event.key == pg.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pg.K_TAB:
                    self.control_all = not self.control_all
                elif event.key in (pg.K_1, pg.K_2, pg.K_3, pg.K_4):
                    self.selected_agent = event.key - pg.K_1

        keys = pg.key.get_pressed()
        action = np.zeros((4, 2), dtype=np.float32)

        selected = self._direction_from_keys(keys, pg.K_LEFT, pg.K_RIGHT, pg.K_UP, pg.K_DOWN)
        action[self.selected_agent] = selected

        if self.control_all:
            action[0] = self._direction_from_keys(keys, pg.K_a, pg.K_d, pg.K_w, pg.K_s)
            action[1] = self._direction_from_keys(keys, pg.K_j, pg.K_l, pg.K_i, pg.K_k)
            action[2] = self._direction_from_keys(keys, pg.K_LEFT, pg.K_RIGHT, pg.K_UP, pg.K_DOWN)
            action[3] = self._direction_from_keys(keys, pg.K_KP4, pg.K_KP6, pg.K_KP8, pg.K_KP5)

        return action * self.force

    @staticmethod
    def _direction_from_keys(keys: object, left: int, right: int, up: int, down: int) -> np.ndarray:
        x = float(keys[right]) - float(keys[left])
        y = float(keys[up]) - float(keys[down])
        vec = np.array([x, y], dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 1.0:
            vec /= norm
        return vec

    def _render(self, info: dict[str, object]) -> None:
        render_info = dict(info)
        render_info["team0_return"] = float(self.team_returns[0])
        render_info["team1_return"] = float(self.team_returns[1])
        render_info["manual_help"] = self._help_text()
        render_info["selected_agent"] = self.selected_agent
        self.renderer.render(render_info)

    def _help_text(self) -> str:
        if self.control_all:
            return "all: P1 WASD | P2 IJKL | P3 arrows | P4 num 8456 | Tab mode | R reset | Space pause"
        return f"selected P{self.selected_agent + 1}: arrows | 1-4 select | Tab all-controls | R reset | Space pause"


def main() -> None:
    args = parse_args()
    env = Football2v2Env(FootballConfig(max_steps=args.max_steps), seed=args.seed)
    renderer = PygameFootballRenderer(env, RenderConfig(fps=args.fps, render_every=1, enabled=True))
    controller = ManualController(env, renderer, force=args.force, control_all=args.control_all)
    controller.run()


if __name__ == "__main__":
    main()
