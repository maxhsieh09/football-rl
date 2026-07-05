from __future__ import annotations

from dataclasses import dataclass
import os
import warnings

from football_rl.env import Football2v2Env


@dataclass
class RenderConfig:
    width: int = 960
    height: int = 620
    margin: int = 44
    fps: int = 60
    render_every: int = 1
    enabled: bool = True


class PygameFootballRenderer:
    def __init__(self, env: Football2v2Env, config: RenderConfig | None = None):
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="pkg_resources is deprecated.*", category=UserWarning)
            import pygame

        self.env = env
        self.cfg = config or RenderConfig()
        self.pygame = pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.cfg.width, self.cfg.height))
        pygame.display.set_caption("2v2 Football Self-Play")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 16)
        self.frame = 0
        self.closed = False

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled and not self.closed

    def render(self, info: dict[str, object] | None = None) -> None:
        self._handle_events()
        if self.closed:
            return

        self.frame += 1
        if not self.cfg.enabled:
            self._draw_paused()
            return
        if self.frame % max(1, self.cfg.render_every) != 0:
            return

        pg = self.pygame
        self.screen.fill((22, 92, 58))
        self._draw_field()
        self._draw_players()
        self._draw_ball()
        self._draw_hud(info or {})
        pg.display.flip()
        self.clock.tick(self.cfg.fps)

    def close(self) -> None:
        if self.closed:
            return
        self.pygame.display.quit()
        self.closed = True

    def _handle_events(self) -> None:
        pg = self.pygame
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.close()
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self.close()
                elif event.key == pg.K_v:
                    self.cfg.enabled = not self.cfg.enabled

    def _draw_field(self) -> None:
        pg = self.pygame
        left, top, width, height = self._field_rect()
        line = (223, 240, 226)
        pg.draw.rect(self.screen, line, (left, top, width, height), 2)
        pg.draw.line(self.screen, line, (left + width / 2, top), (left + width / 2, top + height), 1)
        pg.draw.circle(self.screen, line, (int(left + width / 2), int(top + height / 2)), int(height * 0.14), 1)

        goal_h = self.env.cfg.goal_width * self._scale()
        goal_top = top + height / 2 - goal_h / 2
        pg.draw.rect(self.screen, (245, 245, 245), (left - 8, goal_top, 8, goal_h), 2)
        pg.draw.rect(self.screen, (245, 245, 245), (left + width, goal_top, 8, goal_h), 2)

    def _draw_players(self) -> None:
        pg = self.pygame
        colors = [(70, 150, 255), (70, 150, 255), (255, 92, 86), (255, 92, 86)]
        outlines = [(16, 66, 125), (16, 66, 125), (132, 36, 34), (132, 36, 34)]
        radius = max(6, int(self.env.cfg.player_radius * self._scale()))
        for idx, pos in enumerate(self.env.player_pos):
            screen_pos = self._world_to_screen(float(pos[0]), float(pos[1]))
            pg.draw.circle(self.screen, outlines[idx], screen_pos, radius + 2)
            pg.draw.circle(self.screen, colors[idx], screen_pos, radius)
            label = self.font.render(str(idx % 2 + 1), True, (255, 255, 255))
            rect = label.get_rect(center=screen_pos)
            self.screen.blit(label, rect)

    def _draw_ball(self) -> None:
        pg = self.pygame
        radius = max(4, int(self.env.cfg.ball_radius * self._scale()))
        screen_pos = self._world_to_screen(float(self.env.ball_pos[0]), float(self.env.ball_pos[1]))
        pg.draw.circle(self.screen, (42, 42, 42), screen_pos, radius + 2)
        pg.draw.circle(self.screen, (250, 250, 238), screen_pos, radius)

    def _draw_hud(self, info: dict[str, object]) -> None:
        team0 = float(info.get("team0_return", 0.0))
        team1 = float(info.get("team1_return", 0.0))
        lines = [
            f"step {self.env.steps} | event {info.get('event')} | V toggle | Esc close",
            f"blue cumulative reward {team0:.3f} | red cumulative reward {team1:.3f}",
        ]
        if "manual_help" in info:
            lines.append(str(info["manual_help"]))
        if "selected_agent" in info:
            lines.append(f"selected player {int(info['selected_agent']) + 1}")
        for row, text in enumerate(lines):
            label = self.font.render(text, True, (245, 245, 245))
            self.screen.blit(label, (12, 12 + row * 20))

    def _draw_paused(self) -> None:
        pg = self.pygame
        self.screen.fill((24, 28, 32))
        label = self.font.render("Visualization paused. Press V to resume or Esc to close.", True, (245, 245, 245))
        rect = label.get_rect(center=(self.cfg.width // 2, self.cfg.height // 2))
        self.screen.blit(label, rect)
        pg.display.flip()
        self.clock.tick(12)

    def _world_to_screen(self, x: float, y: float) -> tuple[int, int]:
        left, top, width, height = self._field_rect()
        sx = left + (x + self.env.cfg.field_length / 2) / self.env.cfg.field_length * width
        sy = top + (self.env.cfg.field_width / 2 - y) / self.env.cfg.field_width * height
        return int(sx), int(sy)

    def _field_rect(self) -> tuple[float, float, float, float]:
        available_w = self.cfg.width - 2 * self.cfg.margin
        available_h = self.cfg.height - 2 * self.cfg.margin
        field_aspect = self.env.cfg.field_length / self.env.cfg.field_width
        if available_w / available_h > field_aspect:
            height = available_h
            width = height * field_aspect
        else:
            width = available_w
            height = width / field_aspect
        left = (self.cfg.width - width) / 2
        top = (self.cfg.height - height) / 2
        return left, top, width, height

    def _scale(self) -> float:
        _, _, width, _ = self._field_rect()
        return width / self.env.cfg.field_length
