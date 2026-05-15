from __future__ import annotations
from typing import Callable

import pygame

from .entities import GRID_SIZE, MAX_TICKS
from .simulator import Simulator


CELL_PX: int = 12
GRID_PX: int = GRID_SIZE * CELL_PX
PANEL_PX: int = 240
WINDOW_W: int = GRID_PX + PANEL_PX
WINDOW_H: int = GRID_PX

BG = (24, 26, 30)
GRID_LINE = (40, 44, 50)
PANEL_BG = (16, 18, 22)
TEXT = (220, 220, 220)
TEXT_DIM = (140, 140, 140)
ACCENT = (250, 200, 100)

BUILDING_COLORS: dict[str, tuple[int, int, int]] = {
    "townhall": (220, 180, 50),
    "cannon":   (210, 80,  60),
    "storage":  (80,  140, 210),
    "wall":     (125, 125, 135),
}
TROOP_COLOR = (255, 150, 80)
RANGE_COLOR = (210, 80, 60, 70)


def render(
    screen: pygame.Surface,
    sim: Simulator,
    font: pygame.font.Font,
    show_ranges: bool = False,
    last_action_text: str = "",
    paused: bool = False,
) -> None:
    screen.fill(BG)

    for i in range(GRID_SIZE + 1):
        pygame.draw.line(screen, GRID_LINE, (i * CELL_PX, 0), (i * CELL_PX, GRID_PX))
        pygame.draw.line(screen, GRID_LINE, (0, i * CELL_PX), (GRID_PX, i * CELL_PX))

    if show_ranges:
        overlay = pygame.Surface((GRID_PX, GRID_PX), pygame.SRCALPHA)
        for b in sim.buildings:
            if b.spec.is_defense and b.alive:
                cx, cy = b.center
                pygame.draw.circle(
                    overlay, RANGE_COLOR,
                    (int(cx * CELL_PX), int(cy * CELL_PX)),
                    int(b.spec.attack_range * CELL_PX), 1,
                )
        screen.blit(overlay, (0, 0))

    for b in sim.buildings:
        if not b.alive:
            continue
        color = BUILDING_COLORS.get(b.spec.kind, (180, 180, 180))
        hp_frac = b.hp / b.spec.hp
        fade = 0.3 + 0.7 * hp_frac
        faded = tuple(int(c * fade) for c in color)
        rect = pygame.Rect(
            b.x * CELL_PX, b.y * CELL_PX,
            b.spec.size * CELL_PX, b.spec.size * CELL_PX,
        )
        pygame.draw.rect(screen, faded, rect)
        pygame.draw.rect(screen, color, rect, 1)
        if b.alive and hp_frac < 1.0:
            bar_w = b.spec.size * CELL_PX
            pygame.draw.rect(screen, (60, 60, 60), (rect.x, rect.y - 5, bar_w, 3))
            pygame.draw.rect(screen, (90, 200, 90), (rect.x, rect.y - 5, int(bar_w * hp_frac), 3))

    for t in sim.active_troops:
        px = int(t.x * CELL_PX)
        py = int(t.y * CELL_PX)
        pygame.draw.circle(screen, TROOP_COLOR, (px, py), 4)
        hp_frac = t.hp / t.spec.hp
        if hp_frac < 1.0:
            pygame.draw.rect(screen, (60, 60, 60), (px - 6, py - 10, 12, 2))
            pygame.draw.rect(screen, (90, 200, 90), (px - 6, py - 10, int(12 * hp_frac), 2))

    panel_x = GRID_PX
    pygame.draw.rect(screen, PANEL_BG, (panel_x, 0, PANEL_PX, GRID_PX))

    lines: list[tuple[str, tuple[int, int, int]]] = []
    if last_action_text:
        lines.append((last_action_text, ACCENT))
        lines.append(("", TEXT))
    lines += [
        (f"Tick: {sim.tick_count}/{MAX_TICKS}", TEXT),
        (f"Damage: {sim.damage_pct * 100:5.1f}%", TEXT),
        (f"Stars: {sim.stars}", TEXT),
        (f"Army left: {sim.army_remaining}", TEXT),
        (f"Active: {len(sim.active_troops)}", TEXT),
        (f"Buildings: {len(sim.alive_buildings)}", TEXT),
        (f"TH: {'destroyed' if sim.townhall_destroyed else 'alive'}", TEXT),
        ("", TEXT),
        ("[PAUSED]" if paused else "[playing]", ACCENT if paused else TEXT_DIM),
        ("", TEXT),
        ("Keys:", TEXT_DIM),
        ("  space  pause/resume", TEXT_DIM),
        ("  right  step 1 tick", TEXT_DIM),
        ("  d      toggle ranges", TEXT_DIM),
        ("  r      reset", TEXT_DIM),
        ("  q/esc  quit", TEXT_DIM),
    ]
    for i, (ln, color) in enumerate(lines):
        screen.blit(font.render(ln, True, color), (panel_x + 12, 12 + i * 18))


def play(
    sim_factory: Callable[[], Simulator],
    fps: int = 30,
    autoplay: bool = True,
    deploy_schedule: list[tuple[int, int, int]] | None = None,
) -> None:
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("CoC RL env")
    font = pygame.font.SysFont("menlo,monaco,courier,monospace", 14)
    clock = pygame.time.Clock()

    sim = sim_factory()
    schedule = sorted(deploy_schedule or [], key=lambda d: d[0])
    schedule_idx = 0

    paused = not autoplay
    show_ranges = False
    step_once = False
    last_action_text = ""
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT and paused:
                    step_once = True
                elif event.key == pygame.K_d:
                    show_ranges = not show_ranges
                elif event.key == pygame.K_r:
                    sim = sim_factory()
                    schedule_idx = 0
                    last_action_text = ""

        while (
            schedule_idx < len(schedule)
            and schedule[schedule_idx][0] <= sim.tick_count
        ):
            _, dx, dy = schedule[schedule_idx]
            ok = sim.deploy(dx, dy)
            last_action_text = f"deploy @ ({dx},{dy}) {'OK' if ok else 'FAIL'}"
            schedule_idx += 1

        if (not paused or step_once) and not sim.is_done:
            sim.tick()
            step_once = False

        render(screen, sim, font, show_ranges, last_action_text, paused=paused)
        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()
