from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import gzip
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image, ImageDraw

from coc_env.entities import DECISION_INTERVAL, GRID_SIZE
from coc_env.env import CoCEnv, N_DEPLOY_ACTIONS, WAIT_ACTION, decode_deploy_action, decode_spell_action
from scripts.render_holdout_grid_evolution import discover_prod_stages
from scripts.render_strategy_evolution import (
    BARB,
    BARB_HEAD,
    BREAKER,
    BREAKER_HEAD,
    BUILDING_COLORS,
    FONT_16,
    FONT_18,
    FONT_20,
    FONT_24,
    FONT_30,
    FONT_38,
    FONT_48,
    Stage,
    deploy_random_action,
    load_json,
    save_json,
)
from scripts.train_maskable_ppo import DEFAULT_SPELL_COMPOSITION, format_composition, import_training_deps, parse_spell_composition


CANVAS_W = 1920
CANVAS_H = 1920
HEADER_H = 72
MARGIN = 28
GAP = 18
TILE = int(min(
    (CANVAS_W - 2 * MARGIN - 3 * GAP) / 4,
    (CANVAS_H - HEADER_H - MARGIN - 3 * GAP) / 4,
))
GRID_W = 4 * TILE + 3 * GAP
GRID_X = int((CANVAS_W - GRID_W) / 2)
GRID_Y = HEADER_H
MAP_PAD = 10
MAP = TILE - 2 * MAP_PAD - 34
BG = (16, 18, 21)
TILE_BG = (25, 28, 33)
MAP_BG = (35, 43, 38)
GRID_LINE = (70, 76, 84)
TEXT = (235, 238, 242)
TEXT_DIM = (165, 173, 184)
MUTED = (116, 125, 137)
GREEN = (67, 171, 109)
BLUE = (92, 150, 245)
YELLOW = (246, 191, 91)
RED = (224, 72, 72)
RANGE = (224, 72, 72)
FIRE = (255, 105, 82)
RAGE = (174, 91, 232)
FREEZE = (96, 204, 244)


@dataclass
class PlaybackFrame:
    tick: int
    damage_pct: float
    score: float
    stars: int
    army_remaining: int
    building_hp: list[float]
    building_revealed: list[bool]
    troops: list[tuple[int, str, float, float, bool]]
    active_spells: list[tuple[str, float, float, float, float]]
    events: list[tuple[float, float, float, float, float]]


@dataclass
class PlaybackRollout:
    seed: int
    max_ticks: int
    buildings: list[dict[str, Any]]
    deployments: list[tuple[int, int, str, float, float]]
    spell_casts: list[tuple[int, str, float, float, float]]
    deaths: list[tuple[int, str, float, float]]
    frames: list[PlaybackFrame]
    final: dict[str, Any]


@dataclass
class PlaybackStage:
    stage: Stage
    rollouts: list[PlaybackRollout]
    summary: dict[str, float | int | str]


def set_thread_defaults() -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, "1")


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, font: Any, fill: tuple[int, int, int] = TEXT) -> None:
    draw.text(xy, value, font=font, fill=fill)


def px(x: float, y: float, ox: int, oy: int, scale: float) -> tuple[int, int]:
    return int(round(ox + x * scale)), int(round(oy + y * scale))


def building_static(env: CoCEnv) -> list[dict[str, Any]]:
    assert env.sim is not None
    return [
        {
            "id": b.id,
            "kind": b.spec.kind,
            "x": b.x,
            "y": b.y,
            "size": b.spec.size,
            "max_hp": float(b.spec.hp),
            "is_defense": bool(b.spec.is_defense),
            "attack_range": float(b.spec.attack_range),
            "min_attack_range": float(b.spec.min_attack_range),
        }
        for b in env.sim.buildings
    ]


def sample_frame(env: CoCEnv, events: list[tuple[float, float, float, float, float]]) -> PlaybackFrame:
    assert env.sim is not None
    return PlaybackFrame(
        tick=int(env.sim.tick_count),
        damage_pct=float(env.sim.damage_pct),
        score=float(env.sim.score),
        stars=int(env.sim.stars),
        army_remaining=int(env.sim.army_remaining),
        building_hp=[max(0.0, float(b.hp)) for b in env.sim.buildings],
        building_revealed=[bool(b.revealed) for b in env.sim.buildings],
        troops=[(t.id, t.spec.kind, float(t.x), float(t.y), bool(t.alive)) for t in env.sim.troops],
        active_spells=[
            (spell.kind, float(spell.x), float(spell.y), float(spell.spec.radius), float(spell.remaining))
            for spell in env.sim.active_spells
        ],
        events=events,
    )


def choose_action(model: Any, obs: dict[str, np.ndarray], mask: np.ndarray, deterministic: bool) -> int:
    action, _ = model.predict(obs, deterministic=deterministic, action_masks=mask)
    return int(np.asarray(action).item())


def simulate_rollout(
    stage: Stage,
    *,
    model: Any | None,
    profile: str,
    seed: int,
    deterministic: bool,
    sample_every_ticks: int,
    spell_composition: dict[str, int],
) -> PlaybackRollout:
    env = CoCEnv(
        layout_profile=profile,
        max_buildings=stage.max_buildings,
        army_composition=stage.army_composition,
        spell_composition=spell_composition,
        max_ticks=stage.max_ticks,
    )
    obs, _ = env.reset(seed=seed, options={"profile": profile})
    assert env.sim is not None

    rng = random.Random(seed + stage.step + 711)
    buildings = building_static(env)
    deployments: list[tuple[int, int, str, float, float]] = []
    spell_casts: list[tuple[int, str, float, float, float]] = []
    deaths: list[tuple[int, str, float, float]] = []
    frames = [sample_frame(env, [])]
    pending_events: list[tuple[float, float, float, float, float]] = []
    last_sample_tick = 0

    while not env.sim.is_done:
        mask = env.action_masks()
        action = deploy_random_action(mask, rng) if model is None else choose_action(model, obs, mask, deterministic)
        if 0 <= action < N_DEPLOY_ACTIONS and bool(mask[action]):
            troop_kind, cx, cy = decode_deploy_action(action)
            troop_id = env.sim.next_troop_id
            if env._can_deploy_cell(cx, cy) and env.sim.deploy(cx + 0.5, cy + 0.5, troop_kind=troop_kind):
                deployments.append((env.sim.tick_count, troop_id, troop_kind, cx + 0.5, cy + 0.5))
        elif WAIT_ACTION < action < env.n_actions and bool(mask[action]):
            spell_kind, cx, cy = decode_spell_action(action)
            if env._can_cast_spell_cell(cx, cy, spell_kind) and env.sim.cast_spell(cx + 0.5, cy + 0.5, spell_kind=spell_kind):
                radius = env.sim.active_spells[-1].spec.radius if env.sim.active_spells else 0.0
                spell_casts.append((env.sim.tick_count, spell_kind, cx + 0.5, cy + 0.5, float(radius)))

        for _ in range(DECISION_INTERVAL):
            if env.sim.is_done:
                break
            alive_before = {
                t.id: (t.spec.kind, float(t.x), float(t.y))
                for t in env.sim.troops
                if t.alive
            }
            env.sim.tick()
            for event in env.sim.visual_events:
                if event.get("kind") in {"defense_fire", "impact", "wall_breaker_explosion"}:
                    pending_events.append((
                        float(event.get("from_x", 0.0)),
                        float(event.get("from_y", 0.0)),
                        float(event.get("to_x", 0.0)),
                        float(event.get("to_y", 0.0)),
                        float(event.get("radius") or event.get("splash_radius") or 0.0),
                    ))
            for tid, (kind, x, y) in alive_before.items():
                troop = next((t for t in env.sim.troops if t.id == tid), None)
                if troop is not None and not troop.alive:
                    deaths.append((env.sim.tick_count, kind, x, y))
            if env.sim.tick_count - last_sample_tick >= sample_every_ticks or env.sim.is_done:
                frames.append(sample_frame(env, pending_events[-40:]))
                pending_events = []
                last_sample_tick = env.sim.tick_count
        obs = env._obs()

    final = {
        "damage_pct": float(env.sim.damage_pct),
        "score": float(env.sim.score),
        "stars": int(env.sim.stars),
        "ticks": int(env.sim.tick_count),
        "deployed": len(deployments),
        "spells_cast": len(spell_casts),
        "rage_casts": sum(1 for _, kind, _, _, _ in spell_casts if kind == "rage"),
        "freeze_casts": sum(1 for _, kind, _, _, _ in spell_casts if kind == "freeze"),
        "army_remaining": int(env.sim.army_remaining),
        "walls_destroyed": int(sum(1 for b in env.sim.buildings if b.spec.kind == "wall" and not b.alive)),
        "defenses_destroyed": int(sum(1 for b in env.sim.buildings if b.spec.is_defense and not b.alive)),
        "terminated": bool(env.sim.is_terminal),
        "truncated": bool(env.sim.is_truncated),
    }
    return PlaybackRollout(seed, stage.max_ticks, buildings, deployments, spell_casts, deaths, frames, final)


def summarize(stage: Stage, rollouts: list[PlaybackRollout]) -> dict[str, float | int | str]:
    damage = np.asarray([r.final["damage_pct"] for r in rollouts], dtype=np.float64)
    score = np.asarray([r.final["score"] for r in rollouts], dtype=np.float64)
    stars = np.asarray([r.final["stars"] for r in rollouts], dtype=np.float64)
    return {
        "label": stage.label,
        "step": stage.step,
        "checkpoint": str(stage.checkpoint) if stage.checkpoint else "random",
        "episodes": len(rollouts),
        "damage_mean": float(damage.mean()),
        "score_mean": float(score.mean()),
        "stars_mean": float(stars.mean()),
        "p_stars_ge_2": float((stars >= 2).mean()),
        "p_damage_ge_50": float((damage >= 0.50).mean()),
        "p_damage_ge_90": float((damage >= 0.90).mean()),
    }


def _stage_worker(task: tuple[int, Stage, list[int], str, str, bool, int, dict[str, int]]) -> tuple[int, PlaybackStage]:
    index, stage, seeds, profile, device, deterministic, sample_every_ticks, spell_composition = task
    set_thread_defaults()
    model = None
    if stage.checkpoint is not None:
        deps = import_training_deps()
        try:
            deps.torch.set_num_threads(1)
        except Exception:
            pass
        model = deps.MaskablePPO.load(str(stage.checkpoint), device=device)
    rollouts = [
        simulate_rollout(
            stage,
            model=model,
            profile=profile,
            seed=seed,
            deterministic=deterministic,
            sample_every_ticks=sample_every_ticks,
            spell_composition=spell_composition,
        )
        for seed in seeds
    ]
    return index, PlaybackStage(stage, rollouts, summarize(stage, rollouts))


def load_cache(path: Path) -> list[PlaybackStage]:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def write_cache(path: Path, stages: list[PlaybackStage]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb", compresslevel=4) as f:
        pickle.dump(stages, f, protocol=pickle.HIGHEST_PROTOCOL)


def resolved_spell_composition(args: argparse.Namespace) -> dict[str, int]:
    value = str(args.spell_composition).strip()
    if value.lower() != "auto":
        return parse_spell_composition(value)
    config = load_json(args.run_dir / "run_config.json")
    configured = config.get("spell_composition")
    if isinstance(configured, dict):
        return {str(kind): int(count) for kind, count in configured.items()}
    return dict(DEFAULT_SPELL_COMPOSITION)


def simulate_or_load(args: argparse.Namespace) -> list[PlaybackStage]:
    if args.cache_path.exists() and not args.refresh_cache:
        print(f"loading cached trajectories: {args.cache_path}", flush=True)
        return load_cache(args.cache_path)

    stages = discover_prod_stages(args.run_dir)
    if args.final_only:
        stages = [stages[-1]]
    elif args.stage_limit is not None:
        stages = [stages[0], *stages[1:1 + args.stage_limit]]
    seeds = [args.seed_start + i for i in range(16)]
    spell_composition = resolved_spell_composition(args)
    print(
        f"simulating trajectories stages={len(stages)} seeds={seeds[0]}..{seeds[-1]} "
        f"workers={args.workers} spells={spell_composition} cache={args.cache_path}",
        flush=True,
    )
    tasks = [
        (idx, stage, seeds, args.profile, args.device, not args.stochastic, args.sample_every_ticks, spell_composition)
        for idx, stage in enumerate(stages)
    ]
    out: list[tuple[int, PlaybackStage]] = []
    if args.workers <= 1:
        for task in tasks:
            idx, stage = _stage_worker(task)
            print_stage_summary(idx + 1, len(tasks), stage)
            out.append((idx, stage))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_stage_worker, task) for task in tasks]
            for future in as_completed(futures):
                idx, stage = future.result()
                print_stage_summary(idx + 1, len(tasks), stage)
                out.append((idx, stage))
    stages_out = [stage for _, stage in sorted(out, key=lambda item: item[0])]
    write_cache(args.cache_path, stages_out)
    write_manifest(args.cache_path.with_suffix(".json"), stages_out, args)
    return stages_out


def print_stage_summary(i: int, total: int, stage: PlaybackStage) -> None:
    s = stage.summary
    print(
        f"[{i}/{total}] {stage.stage.label} "
        f"score={float(s['score_mean']):.3f} damage={float(s['damage_mean']):.3f} "
        f"p2={float(s['p_stars_ge_2']):.2f}",
        flush=True,
    )


def write_manifest(path: Path, stages: list[PlaybackStage], args: argparse.Namespace) -> None:
    save_json(path, {
        "run_dir": str(args.run_dir),
        "profile": args.profile,
        "seed_start": args.seed_start,
        "grid": "4x4",
        "sample_every_ticks": args.sample_every_ticks,
        "spell_composition": resolved_spell_composition(args),
        "stages": [
            {
                "label": stage.stage.label,
                "step": stage.stage.step,
                "checkpoint": str(stage.stage.checkpoint) if stage.stage.checkpoint else None,
                "summary": stage.summary,
                "episodes": [{"seed": r.seed, **r.final} for r in stage.rollouts],
            }
            for stage in stages
        ],
    })


def frame_at(rollout: PlaybackRollout, progress: float) -> tuple[int, PlaybackFrame]:
    if not rollout.frames:
        raise ValueError("rollout has no frames")
    target = progress * max(1, rollout.frames[-1].tick)
    ticks = [frame.tick for frame in rollout.frames]
    idx = int(np.searchsorted(ticks, target, side="left"))
    idx = max(0, min(len(rollout.frames) - 1, idx))
    return idx, rollout.frames[idx]


def draw_tile(
    draw: ImageDraw.ImageDraw,
    rollout: PlaybackRollout,
    frame_index: int,
    frame: PlaybackFrame,
    *,
    ox: int,
    oy: int,
) -> None:
    scale = MAP / GRID_SIZE
    draw.rounded_rectangle((ox, oy, ox + TILE, oy + TILE), radius=10, fill=TILE_BG)
    mx = ox + MAP_PAD
    my = oy + MAP_PAD
    draw.rectangle((mx, my, mx + MAP, my + MAP), fill=MAP_BG, outline=(79, 86, 94), width=1)
    for i in range(0, GRID_SIZE + 1, 4):
        x = int(round(mx + i * scale))
        y = int(round(my + i * scale))
        draw.line((x, my, x, my + MAP), fill=GRID_LINE, width=1)
        draw.line((mx, y, mx + MAP, y), fill=GRID_LINE, width=1)

    for i, b in enumerate(rollout.buildings):
        if not frame.building_revealed[i]:
            continue
        hp = frame.building_hp[i]
        alive = hp > 0.0
        kind = str(b["kind"])
        if b["is_defense"] and alive:
            cx = float(b["x"]) + float(b["size"]) / 2.0
            cy = float(b["y"]) + float(b["size"]) / 2.0
            x, y = px(cx, cy, mx, my, scale)
            r = int(float(b["attack_range"]) * scale)
            draw.ellipse((x - r, y - r, x + r, y + r), outline=RANGE, width=1)

    for kind, x, y, radius, remaining in getattr(frame, "active_spells", []):
        cx, cy = px(x, y, mx, my, scale)
        r = int(radius * scale)
        color = FREEZE if kind == "freeze" else RAGE
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color, width=3)
        draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=color)
        label = "F" if kind == "freeze" else "R"
        text(draw, (cx + 6, cy - 11), label, FONT_16, color)

    for i, b in enumerate(rollout.buildings):
        if not frame.building_revealed[i]:
            continue
        hp = frame.building_hp[i]
        alive = hp > 0.0
        kind = str(b["kind"])
        if not alive and kind == "wall":
            continue
        color = BUILDING_COLORS.get(kind, (190, 195, 200))
        if not alive:
            color = tuple(max(30, int(c * 0.25)) for c in color)
        else:
            frac = max(0.0, min(1.0, hp / max(1.0, float(b["max_hp"]))))
            color = tuple(int(c * (0.45 + 0.55 * frac)) for c in color)
        x0, y0 = px(float(b["x"]), float(b["y"]), mx, my, scale)
        x1, y1 = px(float(b["x"]) + float(b["size"]), float(b["y"]) + float(b["size"]), mx, my, scale)
        draw.rectangle((x0, y0, x1, y1), fill=color, outline=color, width=1)

    tracks: dict[int, list[tuple[float, float, str]]] = {}
    for sample in rollout.frames[:frame_index + 1]:
        for tid, kind, x, y, alive in sample.troops:
            if alive:
                tracks.setdefault(tid, []).append((x, y, kind))
    for points in tracks.values():
        if len(points) < 2:
            continue
        kind = points[-1][2]
        color = BREAKER if kind == "wall_breaker" else BARB
        coords = [px(x, y, mx, my, scale) for x, y, _ in points]
        draw.line(coords, fill=color[:3], width=2 if kind == "wall_breaker" else 1)

    for tick, _, kind, x, y in rollout.deployments:
        if tick > frame.tick:
            continue
        color = BREAKER_HEAD if kind == "wall_breaker" else BARB_HEAD
        cx, cy = px(x, y, mx, my, scale)
        draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=color)

    for tick, kind, x, y, radius in getattr(rollout, "spell_casts", []):
        if tick > frame.tick:
            continue
        color = FREEZE if kind == "freeze" else RAGE
        cx, cy = px(x, y, mx, my, scale)
        draw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill=color)

    for tick, kind, x, y in rollout.deaths:
        if tick > frame.tick:
            continue
        cx, cy = px(x, y, mx, my, scale)
        draw.line((cx - 3, cy - 3, cx + 3, cy + 3), fill=(230, 230, 230), width=1)
        draw.line((cx - 3, cy + 3, cx + 3, cy - 3), fill=(230, 230, 230), width=1)

    for fx, fy, tx, ty, radius in frame.events[-18:]:
        x0, y0 = px(fx, fy, mx, my, scale)
        x1, y1 = px(tx, ty, mx, my, scale)
        draw.line((x0, y0, x1, y1), fill=FIRE, width=1)
        if radius > 0:
            r = int(radius * scale)
            draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r), outline=FIRE, width=1)

    for _, kind, x, y, alive in frame.troops:
        if not alive:
            continue
        color = BREAKER_HEAD if kind == "wall_breaker" else BARB_HEAD
        cx, cy = px(x, y, mx, my, scale)
        draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=color, outline=(10, 12, 15), width=1)

    caption_y = oy + MAP_PAD + MAP + 8
    casts = [cast for cast in getattr(rollout, "spell_casts", []) if cast[0] <= frame.tick]
    rage_casts = sum(1 for _, kind, _, _, _ in casts if kind == "rage")
    freeze_casts = sum(1 for _, kind, _, _, _ in casts if kind == "freeze")
    caption = f"s{rollout.seed} | {frame.stars}★ {frame.damage_pct * 100:4.0f}% | t{frame.tick} | R{rage_casts} F{freeze_casts}"
    text(draw, (ox + MAP_PAD + 2, caption_y), caption, FONT_16, TEXT_DIM)


def draw_header(draw: ImageDraw.ImageDraw, stage: PlaybackStage, stage_index: int, total_stages: int, progress: float) -> None:
    s = stage.summary
    draw.rectangle((0, 0, CANVAS_W, HEADER_H), fill=BG)
    stage_color = YELLOW if stage.stage.checkpoint is None else (108, 196, 255)
    prefix = f"ClashAI medium holdouts | {stage_index + 1}/{total_stages} | "
    text(draw, (28, 18), prefix, FONT_20, TEXT_DIM)
    x = 28 + int(draw.textlength(prefix, font=FONT_20))
    text(draw, (x, 18), stage.stage.label, FONT_20, stage_color)
    stat = (
        f" | mean score {float(s['score_mean']):.2f} | "
        f"damage {float(s['damage_mean']) * 100:.1f}% | "
        f"2+star {float(s['p_stars_ge_2']) * 100:.0f}% | "
        f"progress {progress * 100:.0f}% | blue barb, orange breaker, violet rage, cyan freeze"
    )
    text(draw, (x + int(draw.textlength(stage.stage.label, font=FONT_20)), 18), stat, FONT_20, TEXT_DIM)
    draw.rectangle((0, HEADER_H - 5, int(CANVAS_W * ((stage_index + progress) / total_stages)), HEADER_H - 1), fill=(83, 169, 238))


def render_playback_frame(stage: PlaybackStage, *, stage_index: int, total_stages: int, progress: float) -> Image.Image:
    image = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(image)
    draw_header(draw, stage, stage_index, total_stages, progress)
    for i, rollout in enumerate(stage.rollouts):
        col = i % 4
        row = i // 4
        ox = GRID_X + col * (TILE + GAP)
        oy = GRID_Y + row * (TILE + GAP)
        idx, frame = frame_at(rollout, progress)
        draw_tile(draw, rollout, idx, frame, ox=ox, oy=oy)
    return image


def render_summary_frame(stages: list[PlaybackStage]) -> Image.Image:
    image = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(image)
    text(draw, (96, 150), "Production Snapshot Summary", FONT_48)
    text(draw, (98, 218), "The same 16 medium holdout bases replayed for random baseline and every production snapshot.", FONT_24, TEXT_DIM)
    values = [(s.stage.label, float(s.summary["score_mean"]), float(s.summary["damage_mean"]), float(s.summary["p_stars_ge_2"])) for s in stages]
    best = max(values, key=lambda row: row[1])
    text(draw, (98, 292), f"Best mean score in this slice: {best[0]} | score {best[1]:.2f} | damage {best[2] * 100:.1f}% | 2+★ {best[3] * 100:.0f}%", FONT_30, TEXT)
    x0, y0, x1, y1 = 110, 420, CANVAS_W - 110, 1300
    draw.rectangle((x0, y0, x1, y1), fill=(25, 28, 34), outline=(70, 76, 86), width=1)
    plot_x0, plot_y0, plot_x1, plot_y1 = x0 + 80, y0 + 90, x1 - 50, y1 - 90
    for frac in (0.25, 0.5, 0.75):
        y = int(plot_y1 - (plot_y1 - plot_y0) * frac)
        draw.line((plot_x0, y, plot_x1, y), fill=(52, 57, 66), width=1)
    coords = []
    for i, (_, score, _, _) in enumerate(values):
        x = plot_x0 + int((plot_x1 - plot_x0) * i / max(1, len(values) - 1))
        y = plot_y1 - int((plot_y1 - plot_y0) * score / 4.0)
        coords.append((x, y))
    draw.line(coords, fill=BLUE, width=4)
    for x, y in coords:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=BLUE)
    text(draw, (x0 + 28, y0 + 28), "Mean score across the 16-base grid", FONT_24, TEXT)
    text(draw, (plot_x0, plot_y1 + 28), "random baseline", FONT_18, TEXT_DIM)
    text(draw, (plot_x1 - 180, plot_y1 + 28), "final model", FONT_18, TEXT_DIM)
    return image


def video_frames(stages: list[PlaybackStage], *, fps: int, seconds_per_stage: float, summary_seconds: float) -> Iterable[Image.Image]:
    frames_per_stage = max(1, int(fps * seconds_per_stage))
    for stage_index, stage in enumerate(stages):
        for frame_idx in range(frames_per_stage):
            progress = frame_idx / max(1, frames_per_stage - 1)
            yield render_playback_frame(stage, stage_index=stage_index, total_stages=len(stages), progress=progress)

    if summary_seconds > 0:
        summary = render_summary_frame(stages)
        for _ in range(int(fps * summary_seconds)):
            yield summary


def write_video(frames: Iterable[Image.Image], output: Path, *, fps: int, crf: int) -> None:
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render animated 4x4 holdout playback over all production snapshots.")
    parser.add_argument("--run-dir", type=Path, default=Path("checkpoints/maskable_ppo/prod_medium_hard_v1_gpu1_seed46"))
    parser.add_argument("--output", type=Path, default=Path("runs/videos/prod_10m_medium_holdout_4x4_playback.mp4"))
    parser.add_argument("--cache-path", type=Path, default=Path("runs/videos/cache/prod_10m_medium_holdout_4x4_playback.pkl.gz"))
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--profile", default="medium")
    parser.add_argument("--seed-start", type=int, default=210000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--spell-composition", default="auto", help=f"Comma list like rage=2,freeze=2, or auto from run_config. Default training value: {format_composition(DEFAULT_SPELL_COMPOSITION)}.")
    parser.add_argument("--sample-every-ticks", type=int, default=10)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seconds-per-stage", type=float, default=3.0)
    parser.add_argument("--summary-seconds", type=float, default=0.0)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--stage-limit", type=int, default=None)
    parser.add_argument("--final-only", action="store_true")
    return parser


def main() -> None:
    set_thread_defaults()
    args = build_arg_parser().parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.sample_every_ticks <= 0:
        raise ValueError("--sample-every-ticks must be positive")
    stages = simulate_or_load(args)
    write_manifest(args.output.with_suffix(".json"), stages, args)
    print(f"encoding playback video: {args.output}", flush=True)
    write_video(
        video_frames(
            stages,
            fps=args.fps,
            seconds_per_stage=args.seconds_per_stage,
            summary_seconds=args.summary_seconds,
        ),
        args.output,
        fps=args.fps,
        crf=args.crf,
    )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
