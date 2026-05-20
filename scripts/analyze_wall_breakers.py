from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import csv
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from coc_env.env import CoCEnv
from scripts.train_maskable_ppo import (
    DEFAULT_ARMY_COMPOSITION,
    DEFAULT_SPELL_COMPOSITION,
    format_composition,
    import_training_deps,
    max_preset_buildings,
    parse_army_composition,
    parse_spell_composition,
    profile_sequence,
    resolve_model_path,
)


@dataclass
class WallBreakerEpisode:
    profile: str
    seed: int
    steps: int
    ticks: int
    score: float
    damage_pct: float
    stars: int
    terminated: bool
    truncated: bool
    wall_breakers_deployed: int
    wall_breakers_alive_final: int
    wall_breaker_explosions: int
    wall_breakers_dead_without_explosion: int
    wb_wall_damage: float
    wb_unique_walls_damaged: int
    wb_unique_walls_destroyed: int
    wb_fresh_walls_destroyed: int
    wb_predamaged_walls_destroyed: int
    final_walls_damaged: int
    final_walls_destroyed: int
    total_wall_damage: float


def _choose_action(model: Any | None, obs: dict[str, np.ndarray], mask: np.ndarray, rng: np.random.Generator, deterministic: bool) -> int:
    if model is None:
        valid = np.flatnonzero(mask)
        if valid.size == 0:
            raise RuntimeError("action mask has no legal actions")
        return int(rng.choice(valid))
    action, _ = model.predict(obs, deterministic=deterministic, action_masks=mask)
    return int(action)


def _install_probe(sim: Any) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    original = sim._explode_wall_breaker

    def wrapped(troop: Any, target: Any) -> None:
        walls = sim._connected_walls(target, max(0, troop.spec.wall_damage_count))
        before = {wall.id: max(0.0, wall.hp) for wall in walls}
        original(troop, target)
        by_id = {wall.id: wall for wall in sim.buildings}
        damaged: list[int] = []
        destroyed: list[int] = []
        fresh_destroyed = 0
        predamaged_destroyed = 0
        damage = 0.0
        for wall_id, hp_before in before.items():
            wall = by_id[wall_id]
            hp_after = max(0.0, wall.hp)
            delta = max(0.0, hp_before - hp_after)
            if delta <= 1e-9:
                continue
            damaged.append(wall_id)
            damage += delta
            if hp_before > 1e-9 and hp_after <= 1e-9:
                destroyed.append(wall_id)
                if abs(hp_before - wall.spec.hp) <= 1e-9:
                    fresh_destroyed += 1
                else:
                    predamaged_destroyed += 1
        events.append({
            "tick": sim.tick_count,
            "troop_id": troop.id,
            "target_id": target.id,
            "damaged": damaged,
            "destroyed": destroyed,
            "fresh_destroyed": fresh_destroyed,
            "predamaged_destroyed": predamaged_destroyed,
            "damage": damage,
        })

    sim._explode_wall_breaker = wrapped
    return events


def run_episode(
    *,
    model: Any | None,
    profile: str,
    seed: int,
    army_composition: dict[str, int],
    spell_composition: dict[str, int],
    max_ticks: int,
    deterministic: bool,
) -> WallBreakerEpisode:
    env = CoCEnv(
        layout_profile=profile,
        max_buildings=max_preset_buildings(),
        army_composition=army_composition,
        spell_composition=spell_composition,
        max_ticks=max_ticks,
    )
    obs, info = env.reset(seed=seed)
    assert env.sim is not None
    events = _install_probe(env.sim)
    rng = np.random.default_rng(seed)
    steps = 0
    terminated = truncated = False

    while not (terminated or truncated):
        action = _choose_action(model, obs, env.action_masks(), rng, deterministic)
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1

    assert env.sim is not None
    walls = [b for b in env.sim.buildings if b.spec.kind == "wall"]
    wall_damage = {wall.id: max(0.0, wall.spec.hp - max(0.0, wall.hp)) for wall in walls}
    damaged_by_wb = {wall_id for event in events for wall_id in event["damaged"]}
    destroyed_by_wb = {wall_id for event in events for wall_id in event["destroyed"]}
    deployed = army_composition.get("wall_breaker", 0) - env.sim.army_remaining_by_kind.get("wall_breaker", 0)
    alive_final = sum(1 for troop in env.sim.troops if troop.spec.kind == "wall_breaker" and troop.alive)

    return WallBreakerEpisode(
        profile=profile,
        seed=seed,
        steps=steps,
        ticks=int(info["ticks_elapsed"]),
        score=float(info["score"]),
        damage_pct=float(info["damage_pct"]),
        stars=int(info["stars"]),
        terminated=bool(terminated),
        truncated=bool(truncated),
        wall_breakers_deployed=int(deployed),
        wall_breakers_alive_final=int(alive_final),
        wall_breaker_explosions=len(events),
        wall_breakers_dead_without_explosion=max(0, int(deployed) - len(events) - int(alive_final)),
        wb_wall_damage=float(sum(event["damage"] for event in events)),
        wb_unique_walls_damaged=len(damaged_by_wb),
        wb_unique_walls_destroyed=len(destroyed_by_wb),
        wb_fresh_walls_destroyed=int(sum(event["fresh_destroyed"] for event in events)),
        wb_predamaged_walls_destroyed=int(sum(event["predamaged_destroyed"] for event in events)),
        final_walls_damaged=sum(1 for damage in wall_damage.values() if damage > 1e-9),
        final_walls_destroyed=sum(1 for wall in walls if not wall.alive),
        total_wall_damage=float(sum(wall_damage.values())),
    )


def _mean(rows: list[WallBreakerEpisode], field: str) -> float:
    return float(np.mean([getattr(row, field) for row in rows])) if rows else 0.0


def summarize_wall_breakers(rows: list[WallBreakerEpisode]) -> dict[str, Any]:
    if not rows:
        return {"episodes": 0}
    deployed = sum(row.wall_breakers_deployed for row in rows)
    explosions = sum(row.wall_breaker_explosions for row in rows)
    destroyed = sum(row.wb_unique_walls_destroyed for row in rows)
    total_wall_damage = sum(row.total_wall_damage for row in rows)
    wb_wall_damage = sum(row.wb_wall_damage for row in rows)
    return {
        "episodes": len(rows),
        "score_mean": _mean(rows, "score"),
        "damage_mean": _mean(rows, "damage_pct"),
        "stars_mean": _mean(rows, "stars"),
        "wall_breakers_deployed_mean": _mean(rows, "wall_breakers_deployed"),
        "wall_breaker_explosions_mean": _mean(rows, "wall_breaker_explosions"),
        "wall_breakers_dead_without_explosion_mean": _mean(rows, "wall_breakers_dead_without_explosion"),
        "wb_unique_walls_damaged_mean": _mean(rows, "wb_unique_walls_damaged"),
        "wb_unique_walls_destroyed_mean": _mean(rows, "wb_unique_walls_destroyed"),
        "final_walls_destroyed_mean": _mean(rows, "final_walls_destroyed"),
        "episodes_with_wb_deployed_rate": float(np.mean([row.wall_breakers_deployed > 0 for row in rows])),
        "episodes_with_wb_explosion_rate": float(np.mean([row.wall_breaker_explosions > 0 for row in rows])),
        "episodes_with_wb_wall_destroy_rate": float(np.mean([row.wb_unique_walls_destroyed > 0 for row in rows])),
        "explosions_per_deployed_wb": explosions / max(1, deployed),
        "destroyed_walls_per_deployed_wb": destroyed / max(1, deployed),
        "destroyed_walls_per_wb_explosion": destroyed / max(1, explosions),
        "wb_wall_damage_fraction_of_total_wall_damage": wb_wall_damage / max(1e-9, total_wall_damage),
        "wb_fresh_walls_destroyed_total": sum(row.wb_fresh_walls_destroyed for row in rows),
        "wb_predamaged_walls_destroyed_total": sum(row.wb_predamaged_walls_destroyed for row in rows),
    }


def _worker(task: tuple[str | None, list[str], int, int, int, dict[str, int], dict[str, int], int, bool, str]) -> tuple[int, list[WallBreakerEpisode]]:
    checkpoint, profiles, seed_start, episodes, offset, army_composition, spell_composition, max_ticks, deterministic, device = task
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    model = None
    if checkpoint is not None:
        model = import_training_deps().MaskablePPO.load(checkpoint, device=device)
    rows = [
        run_episode(
            model=model,
            profile=profiles[(offset + i) % len(profiles)],
            seed=seed_start + offset + i,
            army_composition=army_composition,
            spell_composition=spell_composition,
            max_ticks=max_ticks,
            deterministic=deterministic,
        )
        for i in range(episodes)
    ]
    return offset, rows


def run_episodes(
    *,
    checkpoint: Path | None,
    profiles: list[str],
    seed_start: int,
    episodes: int,
    army_composition: dict[str, int],
    spell_composition: dict[str, int],
    max_ticks: int,
    deterministic: bool,
    device: str,
    workers: int,
) -> list[WallBreakerEpisode]:
    if workers <= 1:
        model = None
        if checkpoint is not None:
            model = import_training_deps().MaskablePPO.load(str(checkpoint), device=device)
        return [
            run_episode(
                model=model,
                profile=profiles[i % len(profiles)],
                seed=seed_start + i,
                army_composition=army_composition,
                spell_composition=spell_composition,
                max_ticks=max_ticks,
                deterministic=deterministic,
            )
            for i in range(episodes)
        ]

    workers = min(workers, episodes)
    base = episodes // workers
    extra = episodes % workers
    tasks = []
    offset = 0
    for worker_id in range(workers):
        count = base + (1 if worker_id < extra else 0)
        if count <= 0:
            continue
        tasks.append((
            str(checkpoint) if checkpoint is not None else None,
            profiles,
            seed_start,
            count,
            offset,
            army_composition,
            spell_composition,
            max_ticks,
            deterministic,
            device,
        ))
        offset += count

    chunks: list[tuple[int, list[WallBreakerEpisode]]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_worker, task) for task in tasks]
        for future in as_completed(futures):
            chunks.append(future.result())
    rows: list[WallBreakerEpisode] = []
    for _, chunk in sorted(chunks, key=lambda item: item[0]):
        rows.extend(chunk)
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze wall-breaker usefulness in policy or random rollouts.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Model zip or run directory. Omit for random legal actions.")
    parser.add_argument("--profile", default="medium")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=300000)
    parser.add_argument("--army-composition", default=format_composition(DEFAULT_ARMY_COMPOSITION))
    parser.add_argument("--spell-composition", default=format_composition(DEFAULT_SPELL_COMPOSITION))
    parser.add_argument("--max-ticks", type=int, default=900)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/analysis/wall_breakers"))
    parser.add_argument("--name", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    profiles = profile_sequence(args.profile)
    army_composition = parse_army_composition(args.army_composition)
    spell_composition = parse_spell_composition(args.spell_composition)
    checkpoint = resolve_model_path(args.checkpoint) if args.checkpoint is not None else None

    start = time.perf_counter()
    rows = run_episodes(
        checkpoint=checkpoint,
        profiles=profiles,
        seed_start=args.seed_start,
        episodes=args.episodes,
        army_composition=army_composition,
        spell_composition=spell_composition,
        max_ticks=args.max_ticks,
        deterministic=not args.stochastic,
        device=args.device,
        workers=args.workers,
    )
    elapsed = time.perf_counter() - start

    name = args.name or f"{'policy' if checkpoint else 'random'}_{'-'.join(profiles)}_n{args.episodes}_seed{args.seed_start}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = [asdict(row) for row in rows]
    summary = {
        **summarize_wall_breakers(rows),
        "elapsed_seconds": elapsed,
        "episodes_per_min": len(rows) * 60.0 / max(1e-9, elapsed),
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "profiles": profiles,
        "seed_start": args.seed_start,
        "army_composition": army_composition,
        "spell_composition": spell_composition,
        "max_ticks": args.max_ticks,
        "workers": args.workers,
    }
    by_profile = {
        profile: summarize_wall_breakers([row for row in rows if row.profile == profile])
        for profile in sorted(set(profiles))
    }
    json_path = args.output_dir / f"{name}.json"
    csv_path = args.output_dir / f"{name}.csv"
    json_path.write_text(json.dumps({"summary": summary, "by_profile": by_profile, "episodes": records}, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({"summary": summary, "by_profile": by_profile, "json": str(json_path), "csv": str(csv_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
