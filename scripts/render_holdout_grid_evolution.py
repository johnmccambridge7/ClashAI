from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
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
from coc_env.env import CoCEnv, N_DEPLOY_ACTIONS, WAIT_ACTION, decode_deploy_action
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
    final_stage,
    fmt_steps,
    load_json,
    save_json,
    stable_checkpoints,
    stage_from_checkpoint,
)
from scripts.train_maskable_ppo import import_training_deps, max_preset_buildings


CANVAS_W = 1920
CANVAS_H = 1080
BG = (17, 19, 22)
PANEL_BG = (25, 28, 34)
GRID_LINE = (70, 76, 84)
TEXT = (235, 238, 242)
TEXT_DIM = (162, 170, 181)
MUTED = (112, 120, 132)
RANGE = (224, 72, 72, 32)
FIRE = (255, 84, 84, 145)
BAR_BG = (58, 63, 72)
GREEN = (67, 171, 109)
BLUE = (92, 150, 245)
YELLOW = (246, 191, 91)
RED = (220, 89, 89)

GRID_X = 48
GRID_Y = 128
TILE = 226
MAP = 206
PANEL_X = GRID_X + 4 * TILE + 42
PANEL_Y = GRID_Y
PANEL_W = CANVAS_W - PANEL_X - 48
PANEL_H = 4 * TILE - 16


@dataclass
class TileRollout:
    seed: int
    buildings: list[dict[str, Any]]
    tracks: dict[int, list[tuple[float, float, bool, str]]]
    deployments: list[tuple[int, str, float, float]]
    deaths: list[tuple[str, float, float]]
    fires: list[tuple[float, float, float, float, float]]
    final: dict[str, Any]


@dataclass
class GridStage:
    stage: Stage
    rollouts: list[TileRollout]
    summary: dict[str, float | int | str]


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, font: Any, fill: tuple[int, int, int] = TEXT) -> None:
    draw.text(xy, value, font=font, fill=fill)


def draw_bar(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, frac: float, color: tuple[int, int, int], label: str) -> None:
    frac = max(0.0, min(1.0, float(frac)))
    draw.rounded_rectangle((x, y, x + w, y + h), radius=6, fill=BAR_BG)
    draw.rounded_rectangle((x, y, x + int(w * frac), y + h), radius=6, fill=color)
    text(draw, (x + 10, y + 3), label, FONT_16, TEXT)


def px(x: float, y: float, ox: int, oy: int, scale: float) -> tuple[int, int]:
    return int(round(ox + x * scale)), int(round(oy + y * scale))


def building_snapshot(env: CoCEnv) -> list[dict[str, Any]]:
    assert env.sim is not None
    return [
        {
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
            "attack_range": float(b.spec.attack_range),
            "min_attack_range": float(b.spec.min_attack_range),
        }
        for b in env.sim.buildings
    ]


def choose_action(model: Any, obs: dict[str, np.ndarray], mask: np.ndarray, deterministic: bool) -> int:
    action, _ = model.predict(obs, deterministic=deterministic, action_masks=mask)
    return int(np.asarray(action).item())


def simulate_tile(
    stage: Stage,
    *,
    model: Any | None,
    profile: str,
    seed: int,
    deterministic: bool,
) -> TileRollout:
    env = CoCEnv(
        layout_profile=profile,
        max_buildings=stage.max_buildings,
        army_composition=stage.army_composition,
        max_ticks=stage.max_ticks,
    )
    obs, _ = env.reset(seed=seed, options={"profile": profile})
    assert env.sim is not None

    rng = random.Random(seed + stage.step + 991)
    tracks: dict[int, list[tuple[float, float, bool, str]]] = {}
    deployments: list[tuple[int, str, float, float]] = []
    deaths: list[tuple[str, float, float]] = []
    fires: list[tuple[float, float, float, float, float]] = []

    while not env.sim.is_done:
        mask = env.action_masks()
        action = deploy_random_action(mask, rng) if model is None else choose_action(model, obs, mask, deterministic)
        if 0 <= action < N_DEPLOY_ACTIONS and mask[action]:
            troop_kind, cx, cy = decode_deploy_action(action)
            troop_id = env.sim.next_troop_id
            if env._can_deploy_cell(cx, cy) and env.sim.deploy(cx + 0.5, cy + 0.5, troop_kind=troop_kind):
                deployments.append((troop_id, troop_kind, cx + 0.5, cy + 0.5))

        for _ in range(DECISION_INTERVAL):
            if env.sim.is_done:
                break
            alive_before = {
                t.id: (float(t.x), float(t.y), t.spec.kind)
                for t in env.sim.troops
                if t.alive
            }
            env.sim.tick()
            for event in env.sim.visual_events:
                kind = event.get("kind")
                if kind in {"defense_fire", "impact", "wall_breaker_explosion"}:
                    fires.append((
                        float(event.get("from_x", 0.0)),
                        float(event.get("from_y", 0.0)),
                        float(event.get("to_x", 0.0)),
                        float(event.get("to_y", 0.0)),
                        float(event.get("radius") or event.get("splash_radius") or 0.0),
                    ))
            for t in env.sim.troops:
                tracks.setdefault(t.id, []).append((float(t.x), float(t.y), bool(t.alive), t.spec.kind))
            for tid, (x, y, kind) in alive_before.items():
                troop = next((t for t in env.sim.troops if t.id == tid), None)
                if troop is not None and not troop.alive:
                    deaths.append((kind, x, y))
        obs = env._obs()

    final = {
        "damage_pct": float(env.sim.damage_pct),
        "score": float(env.sim.score),
        "stars": int(env.sim.stars),
        "ticks": int(env.sim.tick_count),
        "deployed": len(deployments),
        "army_remaining": int(env.sim.army_remaining),
        "walls_destroyed": int(sum(1 for b in env.sim.buildings if b.spec.kind == "wall" and not b.alive)),
        "defenses_destroyed": int(sum(1 for b in env.sim.buildings if b.spec.is_defense and not b.alive)),
        "terminated": bool(env.sim.is_terminal),
        "truncated": bool(env.sim.is_truncated),
    }
    return TileRollout(seed, building_snapshot(env), tracks, deployments, deaths, fires[-80:], final)


def summarize_stage(stage: Stage, rollouts: list[TileRollout]) -> dict[str, float | int | str]:
    damage = np.array([r.final["damage_pct"] for r in rollouts], dtype=np.float64)
    score = np.array([r.final["score"] for r in rollouts], dtype=np.float64)
    stars = np.array([r.final["stars"] for r in rollouts], dtype=np.float64)
    return {
        "label": stage.label,
        "step": stage.step,
        "checkpoint": str(stage.checkpoint) if stage.checkpoint else "random",
        "damage_mean": float(damage.mean()),
        "damage_median": float(np.median(damage)),
        "score_mean": float(score.mean()),
        "score_median": float(np.median(score)),
        "stars_mean": float(stars.mean()),
        "p_stars_ge_2": float((stars >= 2).mean()),
        "p_damage_ge_50": float((damage >= 0.50).mean()),
        "p_damage_ge_90": float((damage >= 0.90).mean()),
        "walls_destroyed_mean": float(np.mean([r.final["walls_destroyed"] for r in rollouts])),
        "defenses_destroyed_mean": float(np.mean([r.final["defenses_destroyed"] for r in rollouts])),
        "episodes": len(rollouts),
    }


def discover_prod_stages(run_dir: Path) -> list[Stage]:
    config = load_json(run_dir / "run_config.json")
    checkpoints = [stage_from_checkpoint(run_dir, p, config) for p in stable_checkpoints(run_dir)]
    final = final_stage(run_dir, config)
    stages = sorted(checkpoints, key=lambda s: s.step)
    if final is not None:
        stages.append(final)
    if not stages:
        raise FileNotFoundError(f"no model snapshots under {run_dir}")
    first = stages[0]
    random_stage = Stage(
        label="random valid deploy",
        checkpoint=None,
        step=0,
        run_name="random",
        army_composition=first.army_composition,
        max_ticks=first.max_ticks,
        max_buildings=first.max_buildings,
        phase="baseline",
    )
    return [random_stage, *stages]


def draw_tile(draw: ImageDraw.ImageDraw, rollout: TileRollout, box: tuple[int, int, int, int]) -> None:
    ox, oy, w, h = box
    scale = MAP / GRID_SIZE
    draw.rounded_rectangle((ox - 5, oy - 5, ox + MAP + 5, oy + MAP + 31), radius=7, fill=(28, 31, 36))
    draw.rectangle((ox, oy, ox + MAP, oy + MAP), fill=(35, 43, 38), outline=(80, 87, 96), width=1)

    for i in range(0, GRID_SIZE + 1, 4):
        x = int(round(ox + i * scale))
        y = int(round(oy + i * scale))
        draw.line((x, oy, x, oy + MAP), fill=GRID_LINE, width=1)
        draw.line((ox, y, ox + MAP, y), fill=GRID_LINE, width=1)

    for b in rollout.buildings:
        if not b["revealed"]:
            continue
        if b["is_defense"] and b["alive"]:
            cx = float(b["x"]) + float(b["size"]) / 2.0
            cy = float(b["y"]) + float(b["size"]) / 2.0
            x, y = px(cx, cy, ox, oy, scale)
            r = int(float(b["attack_range"]) * scale)
            draw.ellipse((x - r, y - r, x + r, y + r), outline=RANGE[:3], width=1)

    for b in rollout.buildings:
        if not b["revealed"]:
            continue
        kind = str(b["kind"])
        color = BUILDING_COLORS.get(kind, (190, 195, 200))
        if not b["alive"]:
            if kind == "wall":
                continue
            color = tuple(max(28, int(c * 0.28)) for c in color)
        x0, y0 = px(float(b["x"]), float(b["y"]), ox, oy, scale)
        x1, y1 = px(float(b["x"]) + float(b["size"]), float(b["y"]) + float(b["size"]), ox, oy, scale)
        width = 1
        draw.rectangle((x0, y0, x1, y1), fill=color, outline=color, width=width)

    for points in rollout.tracks.values():
        if len(points) < 2:
            continue
        kind = points[-1][3]
        color = BREAKER if kind == "wall_breaker" else BARB
        coords = [px(x, y, ox, oy, scale) for x, y, _, _ in points]
        draw.line(coords, fill=color[:3], width=2 if kind == "wall_breaker" else 1)

    for _, troop_kind, x, y in rollout.deployments:
        color = BREAKER_HEAD if troop_kind == "wall_breaker" else BARB_HEAD
        cx, cy = px(x, y, ox, oy, scale)
        draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=color)

    for kind, x, y in rollout.deaths:
        cx, cy = px(x, y, ox, oy, scale)
        draw.line((cx - 3, cy - 3, cx + 3, cy + 3), fill=(225, 225, 225), width=1)
        draw.line((cx - 3, cy + 3, cx + 3, cy - 3), fill=(225, 225, 225), width=1)

    for fx, fy, tx, ty, radius in rollout.fires:
        x0, y0 = px(fx, fy, ox, oy, scale)
        x1, y1 = px(tx, ty, ox, oy, scale)
        draw.line((x0, y0, x1, y1), fill=FIRE[:3], width=1)
        if radius > 0:
            r = int(radius * scale)
            draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r), outline=(255, 120, 80), width=1)

    final = rollout.final
    caption = f"s{rollout.seed}  {final['stars']}★  {final['damage_pct'] * 100:4.0f}%  {final['score']:.2f}"
    text(draw, (ox + 4, oy + MAP + 8), caption, FONT_16, TEXT_DIM)


def draw_panel(draw: ImageDraw.ImageDraw, grid_stage: GridStage, stage_index: int, total_stages: int) -> None:
    s = grid_stage.summary
    stage = grid_stage.stage
    draw.rounded_rectangle((PANEL_X, PANEL_Y - 5, PANEL_X + PANEL_W, PANEL_Y + PANEL_H), radius=10, fill=PANEL_BG)
    y = PANEL_Y + 28
    text(draw, (PANEL_X + 28, y), "4x4 Holdout Strategy Grid", FONT_38)
    y += 54
    text(draw, (PANEL_X + 28, y), f"Stage {stage_index + 1}/{total_stages}", FONT_20, TEXT_DIM)
    y += 30
    color = YELLOW if stage.checkpoint is None else (108, 196, 255)
    text(draw, (PANEL_X + 28, y), stage.label, FONT_30, color)
    y += 38
    text(draw, (PANEL_X + 28, y), stage.run_name.replace("_", " "), FONT_18, TEXT_DIM)
    y += 48
    draw_bar(draw, PANEL_X + 28, y, PANEL_W - 56, 28, float(s["damage_mean"]), GREEN, f"Mean damage {float(s['damage_mean']) * 100:5.1f}%")
    y += 46
    draw_bar(draw, PANEL_X + 28, y, PANEL_W - 56, 28, float(s["score_mean"]) / 4.0, BLUE, f"Mean score {float(s['score_mean']):4.2f} / 4")
    y += 46
    draw_bar(draw, PANEL_X + 28, y, PANEL_W - 56, 28, float(s["p_stars_ge_2"]), YELLOW, f"2+ star rate {float(s['p_stars_ge_2']) * 100:4.0f}%")
    y += 62

    rows = [
        ("Episodes", str(s["episodes"])),
        ("Mean stars", f"{float(s['stars_mean']):.2f}"),
        ("Median damage", f"{float(s['damage_median']) * 100:.1f}%"),
        ("Median score", f"{float(s['score_median']):.2f}"),
        ("P(damage >= 50%)", f"{float(s['p_damage_ge_50']) * 100:.0f}%"),
        ("P(damage >= 90%)", f"{float(s['p_damage_ge_90']) * 100:.0f}%"),
        ("Mean walls destroyed", f"{float(s['walls_destroyed_mean']):.1f}"),
        ("Mean defenses destroyed", f"{float(s['defenses_destroyed_mean']):.1f}"),
    ]
    for name, value in rows:
        text(draw, (PANEL_X + 28, y), name, FONT_20, TEXT_DIM)
        text(draw, (PANEL_X + PANEL_W - 190, y), value, FONT_20, TEXT)
        y += 31

    y += 24
    text(draw, (PANEL_X + 28, y), "Reading The Grid", FONT_24)
    y += 34
    legend = [
        (BARB_HEAD, "blue: barbarian deploy/path"),
        (BREAKER_HEAD, "orange: wall breaker deploy/path"),
        (RED, "red: defense fire / splash"),
        ((225, 225, 225), "x: troop death"),
    ]
    for color, label in legend:
        draw.ellipse((PANEL_X + 31, y + 6, PANEL_X + 43, y + 18), fill=color)
        text(draw, (PANEL_X + 58, y), label, FONT_18, TEXT_DIM)
        y += 27

    progress_w = PANEL_W - 56
    y = PANEL_Y + PANEL_H - 46
    draw.rounded_rectangle((PANEL_X + 28, y, PANEL_X + 28 + progress_w, y + 9), radius=5, fill=(50, 54, 62))
    draw.rounded_rectangle(
        (PANEL_X + 28, y, PANEL_X + 28 + int(progress_w * ((stage_index + 1) / total_stages)), y + 9),
        radius=5,
        fill=(83, 169, 238),
    )


def render_stage_frame(grid_stage: GridStage, *, stage_index: int, total_stages: int) -> Image.Image:
    image = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(image)
    text(draw, (48, 24), "ClashAI Production Policy Evolution", FONT_48)
    text(draw, (50, 86), "16 fixed hard holdout bases; each frame is the final attack trace for one model snapshot", FONT_20, TEXT_DIM)
    for i, rollout in enumerate(grid_stage.rollouts):
        col = i % 4
        row = i // 4
        x = GRID_X + col * TILE
        y = GRID_Y + row * TILE
        draw_tile(draw, rollout, (x, y, MAP, MAP + 30))
    draw_panel(draw, grid_stage, stage_index, total_stages)
    return image


def line_chart(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    values: list[float],
    *,
    label: str,
    color: tuple[int, int, int],
    ymax: float,
) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=(26, 29, 35), outline=(65, 70, 80), width=1)
    text(draw, (x0 + 12, y0 + 8), label, FONT_20, TEXT)
    if len(values) < 2:
        return
    pad_l, pad_r, pad_t, pad_b = 44, 18, 42, 28
    plot = (x0 + pad_l, y0 + pad_t, x1 - pad_r, y1 - pad_b)
    for frac in (0.25, 0.5, 0.75):
        y = int(plot[3] - (plot[3] - plot[1]) * frac)
        draw.line((plot[0], y, plot[2], y), fill=(48, 52, 60), width=1)
    coords = []
    for i, value in enumerate(values):
        x = plot[0] + int((plot[2] - plot[0]) * i / max(1, len(values) - 1))
        y = plot[3] - int((plot[3] - plot[1]) * max(0.0, min(ymax, value)) / ymax)
        coords.append((x, y))
    draw.line(coords, fill=color, width=3)
    for x, y in coords:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
    text(draw, (x0 + 10, y1 - 24), "random -> final", FONT_16, MUTED)


def render_summary_frame(stages: list[GridStage]) -> Image.Image:
    image = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(image)
    text(draw, (74, 70), "Holdout Evolution Summary", FONT_48)
    text(draw, (76, 132), "Aggregate metrics across the same 16 hard holdout bases for every snapshot.", FONT_24, TEXT_DIM)
    damage = [float(s.summary["damage_mean"]) for s in stages]
    score = [float(s.summary["score_mean"]) for s in stages]
    p2 = [float(s.summary["p_stars_ge_2"]) for s in stages]
    line_chart(draw, (74, 220, 1846, 430), damage, label="Mean Damage", color=GREEN, ymax=1.0)
    line_chart(draw, (74, 470, 1846, 680), score, label="Mean Score", color=BLUE, ymax=4.0)
    line_chart(draw, (74, 720, 1846, 930), p2, label="2+ Star Rate", color=YELLOW, ymax=1.0)
    best = max(stages, key=lambda s: float(s.summary["score_mean"]))
    text(
        draw,
        (76, 972),
        f"Best mean score in this 16-base slice: {best.stage.label} ({float(best.summary['score_mean']):.2f})",
        FONT_24,
        TEXT,
    )
    return image


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


def video_frames(stages: list[GridStage], *, fps: int, seconds_per_stage: float, summary_seconds: float) -> Iterable[Image.Image]:
    intro = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(intro)
    text(draw, (90, 280), "ClashAI 4x4 Holdout Evolution", FONT_48)
    text(draw, (92, 350), "Random baseline, then all 21 completed production model snapshots.", FONT_30, (108, 196, 255))
    text(draw, (92, 410), "Each tile is a different hard holdout base seed.", FONT_24, TEXT)
    text(draw, (92, 448), "Each stage shows final cumulative troop paths and attack outcome.", FONT_24, TEXT_DIM)
    for _ in range(int(fps * 2.0)):
        yield intro

    hold = max(1, int(fps * seconds_per_stage))
    total = len(stages)
    for idx, stage in enumerate(stages):
        frame = render_stage_frame(stage, stage_index=idx, total_stages=total)
        for _ in range(hold):
            yield frame

    summary = render_summary_frame(stages)
    for _ in range(int(fps * summary_seconds)):
        yield summary


def serialize_manifest(stages: list[GridStage], output: Path, args: argparse.Namespace) -> None:
    save_json(output.with_suffix(".json"), {
        "output": str(output),
        "run_dir": str(args.run_dir),
        "profile": args.profile,
        "seed_start": args.seed_start,
        "grid": "4x4",
        "stages": [
            {
                "label": s.stage.label,
                "step": s.stage.step,
                "run_name": s.stage.run_name,
                "checkpoint": str(s.stage.checkpoint) if s.stage.checkpoint else None,
                "summary": s.summary,
                "episodes": [
                    {
                        "seed": r.seed,
                        **r.final,
                    }
                    for r in s.rollouts
                ],
            }
            for s in stages
        ],
    })


def render(args: argparse.Namespace) -> Path:
    stages = discover_prod_stages(args.run_dir)
    if args.stage_limit is not None:
        # Keep random plus the first N model snapshots for smoke tests.
        stages = [stages[0], *stages[1:1 + args.stage_limit]]
    seeds = [args.seed_start + i for i in range(16)]
    deps = import_training_deps()

    grid_stages: list[GridStage] = []
    print(f"stages={len(stages)} seeds={seeds[0]}..{seeds[-1]} output={args.output}", flush=True)
    for idx, stage in enumerate(stages, 1):
        print(f"[{idx}/{len(stages)}] {stage.label} | {stage.checkpoint or 'random'}", flush=True)
        model = deps.MaskablePPO.load(str(stage.checkpoint), device=args.device) if stage.checkpoint is not None else None
        rollouts = [
            simulate_tile(stage, model=model, profile=args.profile, seed=seed, deterministic=not args.stochastic)
            for seed in seeds
        ]
        summary = summarize_stage(stage, rollouts)
        print(
            f"  mean score={float(summary['score_mean']):.3f} "
            f"damage={float(summary['damage_mean']):.3f} "
            f"p2={float(summary['p_stars_ge_2']):.2f}",
            flush=True,
        )
        grid_stages.append(GridStage(stage, rollouts, summary))
        del model

    serialize_manifest(grid_stages, args.output, args)
    write_video(
        video_frames(
            grid_stages,
            fps=args.fps,
            seconds_per_stage=args.seconds_per_stage,
            summary_seconds=args.summary_seconds,
        ),
        args.output,
        fps=args.fps,
        crf=args.crf,
    )
    print(f"wrote {args.output}", flush=True)
    return args.output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a 4x4 holdout-base evolution video for a completed production run.")
    parser.add_argument("--run-dir", type=Path, default=Path("checkpoints/maskable_ppo/prod_medium_hard_v1_gpu1_seed46"))
    parser.add_argument("--output", type=Path, default=Path("runs/videos/prod_10m_holdout_4x4_evolution.mp4"))
    parser.add_argument("--profile", default="hard")
    parser.add_argument("--seed-start", type=int, default=230000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seconds-per-stage", type=float, default=1.6)
    parser.add_argument("--summary-seconds", type=float, default=5.0)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--stage-limit", type=int, default=None, help="Smoke-test limit for model snapshots; random is always included.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.seconds_per_stage <= 0:
        raise ValueError("--seconds-per-stage must be positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.stage_limit is not None and args.stage_limit <= 0:
        raise ValueError("--stage-limit must be positive")
    render(args)


if __name__ == "__main__":
    main()
