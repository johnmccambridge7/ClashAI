from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from coc_env.entities import DECISION_INTERVAL, GRID_SIZE
from coc_env.env import CoCEnv, N_DEPLOY_ACTIONS, WAIT_ACTION, decode_deploy_action
from scripts.train_maskable_ppo import (
    checkpoint_step,
    import_training_deps,
    max_preset_buildings,
)


CANVAS_W = 1920
CANVAS_H = 1080
BASE_X = 54
BASE_Y = 118
BASE_PX = 930
CELL = BASE_PX / GRID_SIZE
PANEL_X = 1030
PANEL_W = CANVAS_W - PANEL_X - 54

BG = (17, 19, 22)
PANEL_BG = (25, 28, 34)
GRID = (67, 72, 80)
TEXT = (235, 238, 242)
TEXT_DIM = (162, 170, 181)
MUTED = (106, 115, 126)
BAR_BG = (58, 63, 72)
BAR_FILL = (67, 171, 109)
BAR_WARN = (228, 166, 70)
BAR_BAD = (220, 89, 89)
RANGE = (224, 72, 72, 34)
BARB = (72, 153, 245, 190)
BREAKER = (250, 160, 60, 210)
BARB_HEAD = (101, 181, 255)
BREAKER_HEAD = (255, 191, 90)
FIRE = (255, 84, 84, 170)

BUILDING_COLORS: dict[str, tuple[int, int, int]] = {
    "townhall": (231, 181, 62),
    "cannon": (216, 82, 74),
    "wizard_tower": (151, 111, 220),
    "mortar": (76, 180, 165),
    "bomb": (118, 102, 86),
    "storage": (80, 139, 214),
    "wall": (136, 141, 151),
}


@dataclass(frozen=True)
class Stage:
    label: str
    checkpoint: Path | None
    step: int
    run_name: str
    army_composition: dict[str, int]
    max_ticks: int
    max_buildings: int
    phase: str


@dataclass
class FrameState:
    tick: int
    damage_pct: float
    score: float
    stars: int
    army_remaining: int
    active_troops: int
    buildings: list[dict[str, Any]]
    troops: list[dict[str, Any]]
    events: list[dict[str, Any]]


@dataclass
class Deployment:
    tick: int
    troop_id: int
    troop_kind: str
    x: float
    y: float
    action_index: int


@dataclass
class Death:
    tick: int
    troop_id: int
    troop_kind: str
    x: float
    y: float


@dataclass
class Rollout:
    stage: Stage
    frames: list[FrameState]
    tracks: dict[int, list[tuple[int, float, float, bool, str]]]
    deployments: list[Deployment]
    troop_deaths: list[Death]
    final: dict[str, Any]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


FONT_16 = font(16)
FONT_18 = font(18)
FONT_20 = font(20)
FONT_24 = font(24)
FONT_30 = font(30, bold=True)
FONT_38 = font(38, bold=True)
FONT_48 = font(48, bold=True)


def fmt_steps(step: int) -> str:
    if step <= 0:
        return "random"
    if step >= 1_000_000:
        return f"{step / 1_000_000:.2f}M steps"
    return f"{step / 1_000:.0f}k steps"


def readable_run_name(run_name: str) -> str:
    return run_name.replace("_", " ")


def stable_checkpoints(run_dir: Path) -> list[Path]:
    checkpoints = sorted(run_dir.glob("checkpoint_*_steps.zip"), key=checkpoint_step)
    return [p for p in checkpoints if p.stat().st_size > 1024]


def stage_from_checkpoint(run_dir: Path, checkpoint: Path, config: dict[str, Any]) -> Stage:
    run_name = str(config.get("run_name", run_dir.name))
    step = checkpoint_step(checkpoint)
    return Stage(
        label=fmt_steps(step),
        checkpoint=checkpoint,
        step=step,
        run_name=run_name,
        army_composition={k: int(v) for k, v in config.get("army_composition", {"barbarian": 40, "wall_breaker": 10}).items()},
        max_ticks=int(config.get("max_ticks", 720)),
        max_buildings=int(config.get("max_buildings", max_preset_buildings())),
        phase=readable_run_name(run_name),
    )


def final_stage(run_dir: Path, config: dict[str, Any]) -> Stage | None:
    checkpoint = run_dir / "final_model.zip"
    if not checkpoint.exists():
        return None
    status_path = run_dir / "run_status.json"
    status = load_json(status_path) if status_path.exists() else {}
    step = int(status.get("num_timesteps") or 0)
    if step <= 0:
        checkpoints = stable_checkpoints(run_dir)
        step = checkpoint_step(checkpoints[-1]) if checkpoints else 0
    return Stage(
        label=f"final ({fmt_steps(step)})",
        checkpoint=checkpoint,
        step=step,
        run_name=str(config.get("run_name", run_dir.name)),
        army_composition={k: int(v) for k, v in config.get("army_composition", {"barbarian": 40, "wall_breaker": 10}).items()},
        max_ticks=int(config.get("max_ticks", 720)),
        max_buildings=int(config.get("max_buildings", max_preset_buildings())),
        phase=readable_run_name(str(config.get("run_name", run_dir.name))),
    )


def select_evenly(items: list[Stage], count: int) -> list[Stage]:
    if len(items) <= count:
        return items
    idxs = np.linspace(0, len(items) - 1, count).round().astype(int).tolist()
    out: list[Stage] = []
    seen: set[int] = set()
    for idx in idxs:
        if idx not in seen:
            out.append(items[idx])
            seen.add(idx)
    if out[-1] is not items[-1]:
        out[-1] = items[-1]
    return out


def discover_stages(run_dirs: Iterable[Path], *, max_policy_stages: int) -> list[Stage]:
    stages: list[Stage] = []
    random_stage: Stage | None = None
    for run_dir in run_dirs:
        config_path = run_dir / "run_config.json"
        if not config_path.exists():
            raise FileNotFoundError(config_path)
        config = load_json(config_path)
        if random_stage is None:
            random_stage = Stage(
                label="random valid deploy",
                checkpoint=None,
                step=0,
                run_name="random",
                army_composition={k: int(v) for k, v in config.get("army_composition", {"barbarian": 40, "wall_breaker": 10}).items()},
                max_ticks=int(config.get("max_ticks", 720)),
                max_buildings=int(config.get("max_buildings", max_preset_buildings())),
                phase="baseline",
            )
        stages.extend(stage_from_checkpoint(run_dir, p, config) for p in stable_checkpoints(run_dir))
        final = final_stage(run_dir, config)
        if final is not None:
            stages.append(final)

    dedup: dict[tuple[str, int, str], Stage] = {}
    for stage in stages:
        key = (stage.run_name, stage.step, str(stage.checkpoint))
        dedup[key] = stage
    policy_stages = sorted(dedup.values(), key=lambda s: (s.step, str(s.checkpoint)))
    selected = select_evenly(policy_stages, max_policy_stages)
    return ([random_stage] if random_stage is not None else []) + selected


def deploy_random_action(mask: np.ndarray, rng: random.Random) -> int:
    valid = [i for i, ok in enumerate(mask) if bool(ok)]
    deploy = [i for i in valid if i != WAIT_ACTION]
    return rng.choice(deploy or valid)


def choose_action(model: Any, obs: dict[str, np.ndarray], mask: np.ndarray, deterministic: bool) -> int:
    action, _ = model.predict(obs, deterministic=deterministic, action_masks=mask)
    return int(np.asarray(action).item())


def building_snapshot(env: CoCEnv) -> list[dict[str, Any]]:
    assert env.sim is not None
    out: list[dict[str, Any]] = []
    for b in env.sim.buildings:
        out.append({
            "id": b.id,
            "kind": b.spec.kind,
            "x": b.x,
            "y": b.y,
            "size": b.spec.size,
            "hp": max(0.0, float(b.hp)),
            "max_hp": float(b.spec.hp),
            "alive": bool(b.alive),
            "revealed": bool(b.revealed),
            "is_defense": bool(b.spec.is_defense),
            "is_trap": bool(b.spec.is_trap),
            "attack_range": float(b.spec.attack_range),
            "min_attack_range": float(b.spec.min_attack_range),
            "splash_radius": float(b.spec.splash_radius),
            "counts_for_score": bool(b.spec.counts_for_score),
        })
    return out


def troop_snapshot(env: CoCEnv) -> list[dict[str, Any]]:
    assert env.sim is not None
    return [
        {
            "id": t.id,
            "kind": t.spec.kind,
            "x": float(t.x),
            "y": float(t.y),
            "hp": max(0.0, float(t.hp)),
            "max_hp": float(t.spec.hp),
            "alive": bool(t.alive),
        }
        for t in env.sim.troops
    ]


def frame_snapshot(env: CoCEnv, events: list[dict[str, Any]]) -> FrameState:
    assert env.sim is not None
    return FrameState(
        tick=int(env.sim.tick_count),
        damage_pct=float(env.sim.damage_pct),
        score=float(env.sim.score),
        stars=int(env.sim.stars),
        army_remaining=int(env.sim.army_remaining),
        active_troops=len(env.sim.active_troops),
        buildings=building_snapshot(env),
        troops=troop_snapshot(env),
        events=events,
    )


def run_rollout(
    stage: Stage,
    *,
    profile: str,
    seed: int,
    record_every_ticks: int,
    deterministic: bool,
    device: str,
) -> Rollout:
    env = CoCEnv(
        layout_profile=profile,
        max_buildings=stage.max_buildings,
        army_composition=stage.army_composition,
        max_ticks=stage.max_ticks,
    )
    obs, _ = env.reset(seed=seed, options={"profile": profile})
    assert env.sim is not None

    model = None
    if stage.checkpoint is not None:
        deps = import_training_deps()
        model = deps.MaskablePPO.load(str(stage.checkpoint), device=device)

    rng = random.Random(seed + stage.step + 17)
    frames = [frame_snapshot(env, [])]
    tracks: dict[int, list[tuple[int, float, float, bool, str]]] = {}
    deployments: list[Deployment] = []
    deaths: list[Death] = []
    last_record_tick = 0
    pending_events: list[dict[str, Any]] = []
    alive_before: dict[int, tuple[float, float, str]] = {}

    while not env.sim.is_done:
        mask = env.action_masks()
        action = deploy_random_action(mask, rng) if model is None else choose_action(model, obs, mask, deterministic)
        if 0 <= action < N_DEPLOY_ACTIONS and mask[action]:
            troop_kind, cx, cy = decode_deploy_action(action)
            troop_id = env.sim.next_troop_id
            if env._can_deploy_cell(cx, cy) and env.sim.deploy(cx + 0.5, cy + 0.5, troop_kind=troop_kind):
                deployments.append(Deployment(env.sim.tick_count, troop_id, troop_kind, cx + 0.5, cy + 0.5, action))

        for _ in range(DECISION_INTERVAL):
            if env.sim.is_done:
                break
            alive_before = {
                t.id: (float(t.x), float(t.y), t.spec.kind)
                for t in env.sim.troops
                if t.alive
            }
            env.sim.tick()
            pending_events.extend(dict(e) for e in env.sim.visual_events)
            for t in env.sim.troops:
                tracks.setdefault(t.id, []).append((env.sim.tick_count, float(t.x), float(t.y), bool(t.alive), t.spec.kind))
            for tid, (x, y, kind) in alive_before.items():
                troop = next((t for t in env.sim.troops if t.id == tid), None)
                if troop is not None and not troop.alive:
                    deaths.append(Death(env.sim.tick_count, tid, kind, x, y))
            if env.sim.tick_count - last_record_tick >= record_every_ticks or env.sim.is_done:
                frames.append(frame_snapshot(env, pending_events))
                pending_events = []
                last_record_tick = env.sim.tick_count
        obs = env._obs()

    final = {
        "tick": env.sim.tick_count,
        "damage_pct": env.sim.damage_pct,
        "score": env.sim.score,
        "stars": env.sim.stars,
        "army_remaining": env.sim.army_remaining,
        "deployed": len(deployments),
        "walls_destroyed": sum(1 for b in env.sim.buildings if b.spec.kind == "wall" and not b.alive),
        "defenses_destroyed": sum(1 for b in env.sim.buildings if b.spec.is_defense and not b.alive),
        "terminated": env.sim.is_terminal,
        "truncated": env.sim.is_truncated,
    }
    return Rollout(stage, frames, tracks, deployments, deaths, final)


def pt(x: float, y: float, scale: float = CELL, ox: float = BASE_X, oy: float = BASE_Y) -> tuple[int, int]:
    return int(round(ox + x * scale)), int(round(oy + y * scale))


def rect_for_building(b: dict[str, Any], scale: float = CELL, ox: float = BASE_X, oy: float = BASE_Y) -> tuple[int, int, int, int]:
    x0, y0 = pt(float(b["x"]), float(b["y"]), scale, ox, oy)
    x1, y1 = pt(float(b["x"]) + float(b["size"]), float(b["y"]) + float(b["size"]), scale, ox, oy)
    return x0, y0, x1, y1


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fnt: ImageFont.FreeTypeFont, fill: tuple[int, int, int] = TEXT) -> None:
    draw.text(xy, text, font=fnt, fill=fill)


def draw_bar(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    h: int,
    frac: float,
    *,
    fill: tuple[int, int, int],
    label: str,
) -> None:
    frac = max(0.0, min(1.0, frac))
    draw.rounded_rectangle((x, y, x + w, y + h), radius=7, fill=BAR_BG)
    draw.rounded_rectangle((x, y, x + int(w * frac), y + h), radius=7, fill=fill)
    draw_text(draw, (x + 10, y + 3), label, FONT_16, TEXT)


def draw_base(draw: ImageDraw.ImageDraw, image: Image.Image, frame: FrameState, rollout: Rollout, *, progress: float) -> None:
    draw.rounded_rectangle((BASE_X - 8, BASE_Y - 8, BASE_X + BASE_PX + 8, BASE_Y + BASE_PX + 8), radius=10, fill=(30, 34, 39))
    draw.rectangle((BASE_X, BASE_Y, BASE_X + BASE_PX, BASE_Y + BASE_PX), fill=(34, 42, 37))
    for i in range(GRID_SIZE + 1):
        p = int(round(BASE_X + i * CELL))
        q = int(round(BASE_Y + i * CELL))
        draw.line((p, BASE_Y, p, BASE_Y + BASE_PX), fill=GRID, width=1)
        draw.line((BASE_X, q, BASE_X + BASE_PX, q), fill=GRID, width=1)

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    for b in frame.buildings:
        if not b["revealed"] or not b["alive"] or not b["is_defense"]:
            continue
        cx = float(b["x"]) + float(b["size"]) / 2.0
        cy = float(b["y"]) + float(b["size"]) / 2.0
        x, y = pt(cx, cy)
        r = int(float(b["attack_range"]) * CELL)
        od.ellipse((x - r, y - r, x + r, y + r), outline=RANGE, width=2)
        min_r = int(float(b["min_attack_range"]) * CELL)
        if min_r > 0:
            od.ellipse((x - min_r, y - min_r, x + min_r, y + min_r), outline=(230, 120, 70, 55), width=1)

    current_tick = frame.tick
    for tid, points in rollout.tracks.items():
        visible = [(x, y, alive, kind) for tick, x, y, alive, kind in points if tick <= current_tick]
        if len(visible) < 2:
            continue
        kind = visible[-1][3]
        color = BREAKER if kind == "wall_breaker" else BARB
        coords = [pt(x, y) for x, y, _, _ in visible]
        if len(coords) > 1:
            od.line(coords, fill=color, width=3 if kind == "wall_breaker" else 2, joint="curve")

    for dep in rollout.deployments:
        if dep.tick > current_tick:
            continue
        x, y = pt(dep.x, dep.y)
        color = BREAKER_HEAD if dep.troop_kind == "wall_breaker" else BARB_HEAD
        od.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(*color, 230), outline=(255, 255, 255, 190), width=1)

    for death in rollout.troop_deaths:
        if death.tick > current_tick:
            continue
        x, y = pt(death.x, death.y)
        od.line((x - 5, y - 5, x + 5, y + 5), fill=(225, 225, 225, 130), width=2)
        od.line((x - 5, y + 5, x + 5, y - 5), fill=(225, 225, 225, 130), width=2)

    for event in frame.events:
        if event.get("kind") not in {"defense_fire", "impact", "wall_breaker_explosion"}:
            continue
        fx, fy = pt(float(event.get("from_x", 0.0)), float(event.get("from_y", 0.0)))
        tx, ty = pt(float(event.get("to_x", 0.0)), float(event.get("to_y", 0.0)))
        od.line((fx, fy, tx, ty), fill=FIRE, width=2)
        radius = float(event.get("radius") or event.get("splash_radius") or 0.0)
        if radius > 0:
            r = int(radius * CELL)
            od.ellipse((tx - r, ty - r, tx + r, ty + r), outline=(255, 120, 80, 120), width=2)
    image.alpha_composite(overlay)

    for b in frame.buildings:
        if not b["revealed"]:
            continue
        x0, y0, x1, y1 = rect_for_building(b)
        kind = str(b["kind"])
        base = BUILDING_COLORS.get(kind, (190, 195, 200))
        if not b["alive"]:
            if kind == "wall":
                continue
            fill = tuple(max(35, int(c * 0.28)) for c in base)
            draw.rectangle((x0, y0, x1, y1), fill=fill, outline=(84, 86, 88), width=1)
            continue
        hp = max(0.0, min(1.0, float(b["hp"]) / max(1.0, float(b["max_hp"]))))
        fill = tuple(int(c * (0.45 + 0.55 * hp)) for c in base)
        width = 1 if kind == "wall" else 2
        draw.rectangle((x0, y0, x1, y1), fill=fill, outline=base, width=width)
        if hp < 0.98 and kind != "wall":
            draw.rectangle((x0, y0 - 5, x1, y0 - 2), fill=(46, 48, 52))
            draw.rectangle((x0, y0 - 5, x0 + int((x1 - x0) * hp), y0 - 2), fill=(96, 205, 124))

    for t in frame.troops:
        if not t["alive"]:
            continue
        x, y = pt(float(t["x"]), float(t["y"]))
        kind = str(t["kind"])
        color = BREAKER_HEAD if kind == "wall_breaker" else BARB_HEAD
        r = 6 if kind == "wall_breaker" else 5
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=(12, 14, 18), width=2)

    draw.rectangle((BASE_X, BASE_Y + BASE_PX + 14, BASE_X + int(BASE_PX * progress), BASE_Y + BASE_PX + 21), fill=(83, 169, 238))
    draw.rectangle((BASE_X, BASE_Y + BASE_PX + 14, BASE_X + BASE_PX, BASE_Y + BASE_PX + 21), outline=(74, 79, 88), width=1)


def draw_panel(draw: ImageDraw.ImageDraw, frame: FrameState, rollout: Rollout, *, title: str) -> None:
    stage = rollout.stage
    draw.rounded_rectangle((PANEL_X, BASE_Y - 8, PANEL_X + PANEL_W, BASE_Y + BASE_PX + 8), radius=10, fill=PANEL_BG)
    y = BASE_Y + 28
    draw_text(draw, (PANEL_X + 28, y), title, FONT_38)
    y += 54
    draw_text(draw, (PANEL_X + 28, y), stage.label, FONT_30, (108, 196, 255) if stage.checkpoint else (246, 191, 91))
    y += 38
    draw_text(draw, (PANEL_X + 28, y), stage.phase, FONT_18, TEXT_DIM)
    y += 48

    draw_bar(draw, PANEL_X + 28, y, PANEL_W - 56, 28, frame.damage_pct, fill=BAR_FILL, label=f"Damage {frame.damage_pct * 100:5.1f}%")
    y += 48
    draw_bar(draw, PANEL_X + 28, y, PANEL_W - 56, 28, frame.stars / 3.0, fill=BAR_WARN, label=f"Stars {frame.stars}/3")
    y += 48
    draw_bar(draw, PANEL_X + 28, y, PANEL_W - 56, 28, frame.score / 4.0, fill=(95, 139, 245), label=f"Score {frame.score:4.2f} / 4")
    y += 64

    walls_destroyed = sum(1 for b in frame.buildings if b["kind"] == "wall" and not b["alive"])
    defenses_total = sum(1 for b in frame.buildings if b["is_defense"])
    defenses_destroyed = sum(1 for b in frame.buildings if b["is_defense"] and not b["alive"])
    breakers_deployed = sum(1 for d in rollout.deployments if d.tick <= frame.tick and d.troop_kind == "wall_breaker")
    barbs_deployed = sum(1 for d in rollout.deployments if d.tick <= frame.tick and d.troop_kind == "barbarian")
    rows = [
        ("Tick", f"{frame.tick}/{stage.max_ticks}"),
        ("Active troops", str(frame.active_troops)),
        ("Army remaining", str(frame.army_remaining)),
        ("Barbarians deployed", str(barbs_deployed)),
        ("Wall breakers deployed", str(breakers_deployed)),
        ("Walls destroyed", str(walls_destroyed)),
        ("Defenses destroyed", f"{defenses_destroyed}/{defenses_total}"),
    ]
    for name, value in rows:
        draw_text(draw, (PANEL_X + 28, y), name, FONT_20, TEXT_DIM)
        draw_text(draw, (PANEL_X + PANEL_W - 220, y), value, FONT_20, TEXT)
        y += 32
    y += 18

    draw_text(draw, (PANEL_X + 28, y), "Encoding", FONT_24, TEXT)
    y += 36
    legend = [
        (BARB_HEAD, "barbarian path / deploy"),
        (BREAKER_HEAD, "wall breaker path / deploy"),
        ((224, 72, 72), "defense range / fire"),
        ((225, 225, 225), "troop death marker"),
    ]
    for color, label in legend:
        draw.ellipse((PANEL_X + 30, y + 5, PANEL_X + 44, y + 19), fill=color)
        draw_text(draw, (PANEL_X + 58, y), label, FONT_18, TEXT_DIM)
        y += 28

    y = BASE_Y + BASE_PX - 114
    final = rollout.final
    draw_text(draw, (PANEL_X + 28, y), "Final outcome for this checkpoint", FONT_24)
    y += 36
    final_line = (
        f"{final['damage_pct'] * 100:.1f}% damage, "
        f"{final['stars']} stars, score {final['score']:.2f}"
    )
    draw_text(draw, (PANEL_X + 28, y), final_line, FONT_20, TEXT)
    y += 30
    draw_text(
        draw,
        (PANEL_X + 28, y),
        f"{final['walls_destroyed']} walls, {final['defenses_destroyed']} defenses destroyed",
        FONT_18,
        TEXT_DIM,
    )


def render_rollout_frame(rollout: Rollout, frame: FrameState, *, title: str) -> Image.Image:
    image = Image.new("RGBA", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(image)
    draw_text(draw, (54, 24), "ClashAI Strategy Evolution", FONT_48)
    draw_text(draw, (56, 86), "fixed hard holdout base, deterministic policy rollout, cumulative troop path traces", FONT_20, TEXT_DIM)
    progress = frame.tick / max(1, rollout.stage.max_ticks)
    draw_base(draw, image, frame, rollout, progress=progress)
    draw_panel(draw, frame, rollout, title=title)
    return image.convert("RGB")


def draw_tracks_small(draw: ImageDraw.ImageDraw, rollout: Rollout, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    size = min(x1 - x0, y1 - y0)
    scale = size / GRID_SIZE
    draw.rectangle((x0, y0, x0 + size, y0 + size), fill=(35, 42, 38), outline=(79, 86, 94), width=1)
    final_frame = rollout.frames[-1]
    for b in final_frame.buildings:
        if not b["revealed"]:
            continue
        bx0, by0, bx1, by1 = rect_for_building(b, scale, x0, y0)
        color = BUILDING_COLORS.get(str(b["kind"]), (180, 180, 180))
        if not b["alive"]:
            color = tuple(int(c * 0.3) for c in color)
        draw.rectangle((bx0, by0, bx1, by1), fill=color, outline=color)
    for points in rollout.tracks.values():
        if len(points) < 2:
            continue
        kind = points[-1][4]
        color = BREAKER if kind == "wall_breaker" else BARB
        coords = [(int(x0 + x * scale), int(y0 + y * scale)) for _, x, y, _, _ in points]
        draw.line(coords, fill=color[:3], width=2 if kind == "wall_breaker" else 1)
    for dep in rollout.deployments:
        color = BREAKER_HEAD if dep.troop_kind == "wall_breaker" else BARB_HEAD
        x = int(x0 + dep.x * scale)
        y = int(y0 + dep.y * scale)
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)


def render_comparison_frame(rollouts: list[Rollout], *, title: str) -> Image.Image:
    image = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(image)
    draw_text(draw, (54, 24), "Strategy Evolution Summary", FONT_48)
    draw_text(draw, (56, 86), title, FONT_20, TEXT_DIM)

    cols = min(4, max(1, math.ceil(math.sqrt(len(rollouts)))))
    rows = math.ceil(len(rollouts) / cols)
    tile_w = (CANVAS_W - 108 - (cols - 1) * 26) // cols
    tile_h = (CANVAS_H - 176 - (rows - 1) * 34) // rows
    map_size = min(tile_w, tile_h - 94)

    for i, rollout in enumerate(rollouts):
        col = i % cols
        row = i // cols
        x = 54 + col * (tile_w + 26)
        y = 154 + row * (tile_h + 34)
        draw.rounded_rectangle((x - 8, y - 8, x + tile_w, y + tile_h), radius=10, fill=PANEL_BG)
        draw_tracks_small(draw, rollout, (x, y, x + map_size, y + map_size))
        fy = y + map_size + 12
        draw_text(draw, (x, fy), rollout.stage.label, FONT_20, TEXT)
        fy += 25
        draw_text(draw, (x, fy), rollout.stage.phase[:38], FONT_16, TEXT_DIM)
        fy += 23
        final = rollout.final
        draw_text(
            draw,
            (x, fy),
            f"{final['damage_pct'] * 100:.1f}% dmg | {final['stars']} stars | score {final['score']:.2f}",
            FONT_16,
            TEXT,
        )
    return image


def write_video(
    frames: Iterable[Image.Image],
    output: Path,
    *,
    fps: int,
    crf: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{CANVAS_W}x{CANVAS_H}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        str(output),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for frame in frames:
            proc.stdin.write(frame.convert("RGB").tobytes())
    finally:
        proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {rc}")


def interpolate_frames(rollout: Rollout, *, frame_count: int) -> list[FrameState]:
    if not rollout.frames:
        return []
    if len(rollout.frames) >= frame_count:
        idxs = np.linspace(0, len(rollout.frames) - 1, frame_count).round().astype(int)
        return [rollout.frames[int(i)] for i in idxs]
    return [rollout.frames[min(len(rollout.frames) - 1, int(round(i)))] for i in np.linspace(0, len(rollout.frames) - 1, frame_count)]


def video_frames(rollouts: list[Rollout], *, fps: int, seconds_per_stage: float, hold_seconds: float, title: str) -> Iterable[Image.Image]:
    intro = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(intro)
    draw_text(draw, (90, 280), "ClashAI Strategy Evolution", FONT_48)
    draw_text(draw, (92, 350), title, FONT_30, (108, 196, 255))
    draw_text(draw, (92, 410), "Each segment replays the same hard holdout base.", FONT_24, TEXT)
    draw_text(draw, (92, 448), "Blue trails are barbarians; orange trails are wall breakers.", FONT_24, TEXT_DIM)
    draw_text(draw, (92, 486), "The final panel compares cumulative deployment and path geometry across checkpoints.", FONT_24, TEXT_DIM)
    for _ in range(int(fps * 2.0)):
        yield intro

    frames_per_stage = max(1, int(fps * seconds_per_stage))
    for rollout in rollouts:
        segment_title = "Random baseline" if rollout.stage.checkpoint is None else "Learned policy checkpoint"
        for frame in interpolate_frames(rollout, frame_count=frames_per_stage):
            yield render_rollout_frame(rollout, frame, title=segment_title)
        final_frame = render_rollout_frame(rollout, rollout.frames[-1], title=segment_title)
        for _ in range(int(fps * hold_seconds)):
            yield final_frame

    comparison = render_comparison_frame(rollouts, title=title)
    for _ in range(int(fps * 5.0)):
        yield comparison


def preset_run_dirs(preset: str) -> list[Path]:
    root = Path("checkpoints/maskable_ppo")
    if preset == "same_army":
        return [
            root / "prod_medium_hard_v1_gpu1_seed46",
            root / "hard_focus_same_army_seed47",
        ]
    if preset == "hard_v2":
        return [root / "hard_v2_more_army_longer_seed48"]
    raise ValueError(f"unknown preset {preset!r}")


def output_for_preset(preset: str, output_dir: Path) -> Path:
    if preset == "same_army":
        return output_dir / "strategy_evolution_same_army.mp4"
    if preset == "hard_v2":
        return output_dir / "strategy_evolution_hard_v2.mp4"
    raise ValueError(f"unknown preset {preset!r}")


def render_one(args: argparse.Namespace, preset: str) -> Path:
    run_dirs = [Path(p) for p in args.run_dir] if args.run_dir else preset_run_dirs(preset)
    stages = discover_stages(run_dirs, max_policy_stages=args.max_policy_stages)
    output = args.output if args.output and args.preset != "both" else output_for_preset(preset, args.output_dir)
    print(f"[{preset}] stages={len(stages)} output={output}", flush=True)
    for stage in stages:
        print(f"  - {stage.label:>18} | {stage.phase} | {stage.checkpoint or 'random'}", flush=True)

    rollouts: list[Rollout] = []
    for i, stage in enumerate(stages, 1):
        print(f"[{preset}] rollout {i}/{len(stages)}: {stage.label}", flush=True)
        rollouts.append(run_rollout(
            stage,
            profile=args.profile,
            seed=args.seed,
            record_every_ticks=args.record_every_ticks,
            deterministic=not args.stochastic,
            device=args.device,
        ))

    manifest = {
        "preset": preset,
        "profile": args.profile,
        "seed": args.seed,
        "output": str(output),
        "stages": [
            {
                "label": rollout.stage.label,
                "step": rollout.stage.step,
                "checkpoint": str(rollout.stage.checkpoint) if rollout.stage.checkpoint else None,
                "run_name": rollout.stage.run_name,
                "phase": rollout.stage.phase,
                "army_composition": rollout.stage.army_composition,
                "max_ticks": rollout.stage.max_ticks,
                "final": rollout.final,
            }
            for rollout in rollouts
        ],
    }
    save_json(output.with_suffix(".json"), manifest)
    print(f"[{preset}] encoding mp4", flush=True)
    write_video(
        video_frames(
            rollouts,
            fps=args.fps,
            seconds_per_stage=args.seconds_per_stage,
            hold_seconds=args.hold_seconds,
            title=args.title or ("same-army warm start plus hard-focus continuation" if preset == "same_army" else "larger-army hard regime from scratch"),
        ),
        output,
        fps=args.fps,
        crf=args.crf,
    )
    print(f"[{preset}] wrote {output}", flush=True)
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render MP4 strategy-evolution videos from MaskablePPO checkpoints.")
    parser.add_argument("--preset", choices=["same_army", "hard_v2", "both"], default="both")
    parser.add_argument("--run-dir", action="append", default=[], help="Override checkpoint run dir; repeatable. Only valid for one preset.")
    parser.add_argument("--output", type=Path, default=None, help="Output MP4 path when rendering one preset.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/videos"))
    parser.add_argument("--profile", default="hard")
    parser.add_argument("--seed", type=int, default=220000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seconds-per-stage", type=float, default=5.0)
    parser.add_argument("--hold-seconds", type=float, default=0.8)
    parser.add_argument("--max-policy-stages", type=int, default=7)
    parser.add_argument("--record-every-ticks", type=int, default=5)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--title", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.preset == "both" and args.run_dir:
        raise ValueError("--run-dir override is only supported with a single preset")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.max_policy_stages <= 0:
        raise ValueError("--max-policy-stages must be positive")
    if args.record_every_ticks <= 0:
        raise ValueError("--record-every-ticks must be positive")

    presets = ["same_army", "hard_v2"] if args.preset == "both" else [args.preset]
    for preset in presets:
        render_one(args, preset)


if __name__ == "__main__":
    main()
