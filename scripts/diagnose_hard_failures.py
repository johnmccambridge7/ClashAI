from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import csv
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from coc_env.entities import DECISION_INTERVAL, GRID_SIZE
from coc_env.env import CoCEnv, N_DEPLOY_ACTIONS, WAIT_ACTION
from scripts.train_maskable_ppo import (
    DEFAULT_ARMY_COMPOSITION,
    DEFAULT_SPELL_COMPOSITION,
    import_training_deps,
    max_preset_buildings,
    parse_army_composition,
    parse_spell_composition,
    profile_sequence,
)


SPLASH_KINDS = {"wizard_tower", "mortar", "bomb"}


@dataclass
class FailureEpisode:
    label: str
    profile: str
    seed: int
    score: float
    damage_pct: float
    stars: int
    steps: int
    ticks: int
    terminated: int
    truncated: int
    deployments: int
    deployment_unique_cells: int
    deployment_top_cell_fraction: float
    deployment_top3_cell_fraction: float
    deployment_quadrant_entropy: float
    active_nearest_distance_mean: float
    active_close_pair_fraction_mean: float
    active_quadrant_entropy_mean: float
    troop_damage_taken_pct: float
    cannon_damage_pct: float
    wizard_tower_damage_pct: float
    mortar_damage_pct: float
    bomb_damage_pct: float
    splash_damage_pct: float
    cannon_deaths: int
    wizard_tower_deaths: int
    mortar_deaths: int
    bomb_deaths: int
    splash_deaths: int
    multi_hit_splash_events: int
    wizard_tower_multi_hit_events: int
    mortar_multi_hit_events: int
    bomb_multi_hit_events: int
    wall_breakers_deployed: int
    wall_breaker_explosions: int
    wall_breakers_dead_without_explosion: int
    wall_segments_damaged_by_wb: int
    wall_segments_destroyed_by_wb: int
    useful_rage_casts: int
    useful_freeze_casts: int
    wasted_spell_casts: int
    spells_remaining: int
    townhall_destroyed: int
    defenses_destroyed: int


def normalized_entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 1:
        return 0.0
    probs = [count / total for count in counts if count > 0]
    return float(-sum(p * math.log(p) for p in probs) / max(1e-9, math.log(min(total, len(counts)))))


def quadrant_index(x: float, y: float) -> int:
    return int(x >= GRID_SIZE / 2.0) + 2 * int(y >= GRID_SIZE / 2.0)


def deployment_metrics(deployments: list[tuple[int, int, str]]) -> dict[str, float]:
    if not deployments:
        return {
            "deployments": 0.0,
            "deployment_unique_cells": 0.0,
            "deployment_top_cell_fraction": 0.0,
            "deployment_top3_cell_fraction": 0.0,
            "deployment_quadrant_entropy": 0.0,
        }
    cell_counts = Counter((x, y) for x, y, _ in deployments)
    ranked = sorted(cell_counts.values(), reverse=True)
    quadrants = [0, 0, 0, 0]
    for x, y, _ in deployments:
        quadrants[quadrant_index(x, y)] += 1
    total = len(deployments)
    return {
        "deployments": float(total),
        "deployment_unique_cells": float(len(cell_counts)),
        "deployment_top_cell_fraction": ranked[0] / total,
        "deployment_top3_cell_fraction": sum(ranked[:3]) / total,
        "deployment_quadrant_entropy": normalized_entropy(quadrants),
    }


def active_spread_metrics(sim: Any) -> tuple[float, float, float]:
    troops = sim.active_troops
    if len(troops) < 2:
        return 0.0, 0.0, 0.0
    nearest: list[float] = []
    close_pairs = 0
    pair_count = 0
    quadrants = [0, 0, 0, 0]
    for troop in troops:
        quadrants[quadrant_index(troop.x, troop.y)] += 1
    for i, troop in enumerate(troops):
        distances = [
            math.hypot(troop.x - other.x, troop.y - other.y)
            for j, other in enumerate(troops)
            if i != j
        ]
        nearest.append(min(distances))
        for other in troops[i + 1:]:
            pair_count += 1
            if math.hypot(troop.x - other.x, troop.y - other.y) <= 1.5:
                close_pairs += 1
    return (
        float(np.mean(nearest)),
        close_pairs / max(1, pair_count),
        normalized_entropy(quadrants),
    )


class EpisodeProbe:
    def __init__(self, sim: Any) -> None:
        self.sim = sim
        self.damage_by_source: defaultdict[str, float] = defaultdict(float)
        self.deaths_by_source: defaultdict[str, int] = defaultdict(int)
        self.multi_hit_by_source: defaultdict[str, int] = defaultdict(int)
        self.wall_breaker_explosions = 0
        self.wall_segments_damaged_by_wb = 0
        self.wall_segments_destroyed_by_wb = 0
        self.active_nearest_samples: list[float] = []
        self.active_close_samples: list[float] = []
        self.active_quadrant_samples: list[float] = []

        original_damage = sim._damage_troops_at
        original_explode = sim._explode_wall_breaker
        original_tick = sim.tick

        def damage_wrapper(
            x: float,
            y: float,
            radius: float,
            damage: float,
            *,
            fallback: Any | None = None,
            source_kind: str | None = None,
        ) -> None:
            hp_before = {troop.id: max(0.0, troop.hp) for troop in sim.troops}
            alive_before = {troop.id for troop in sim.troops if troop.alive}
            original_damage(x, y, radius, damage, fallback=fallback, source_kind=source_kind)
            source = source_kind or "unknown"
            hits = 0
            for troop in sim.troops:
                before = hp_before.get(troop.id, 0.0)
                after = max(0.0, troop.hp)
                delta = max(0.0, before - after)
                if delta <= 0.0:
                    continue
                hits += 1
                self.damage_by_source[source] += delta
                if troop.id in alive_before and after <= 0.0:
                    self.deaths_by_source[source] += 1
            if source in SPLASH_KINDS and hits >= 2:
                self.multi_hit_by_source[source] += 1

        def explode_wrapper(troop: Any, target: Any) -> None:
            wall_hp_before = {
                building.id: max(0.0, building.hp)
                for building in sim.buildings
                if building.alive and building.spec.kind == "wall"
            }
            original_explode(troop, target)
            self.wall_breaker_explosions += 1
            damaged = 0
            destroyed = 0
            for building in sim.buildings:
                if building.spec.kind != "wall" or building.id not in wall_hp_before:
                    continue
                before = wall_hp_before[building.id]
                after = max(0.0, building.hp)
                if after < before:
                    damaged += 1
                if before > 0.0 and after <= 0.0:
                    destroyed += 1
            self.wall_segments_damaged_by_wb += damaged
            self.wall_segments_destroyed_by_wb += destroyed

        def tick_wrapper() -> None:
            if len(sim.active_troops) >= 2:
                nearest, close, quadrant_entropy = active_spread_metrics(sim)
                self.active_nearest_samples.append(nearest)
                self.active_close_samples.append(close)
                self.active_quadrant_samples.append(quadrant_entropy)
            original_tick()

        sim._damage_troops_at = damage_wrapper
        sim._explode_wall_breaker = explode_wrapper
        sim.tick = tick_wrapper


def choose_action(model: Any | None, obs: dict[str, np.ndarray], mask: np.ndarray, rng: np.random.Generator, deterministic: bool) -> int:
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        return WAIT_ACTION
    if model is None:
        return int(rng.choice(legal))
    action, _ = model.predict(obs, deterministic=deterministic, action_masks=mask)
    return int(np.asarray(action).item())


def diagnose_episode(
    *,
    label: str,
    model: Any | None,
    profile: str,
    seed: int,
    max_buildings: int,
    army_composition: dict[str, int],
    spell_composition: dict[str, int],
    max_ticks: int,
    deterministic: bool,
) -> FailureEpisode:
    env = CoCEnv(
        layout_profile=profile,
        max_buildings=max_buildings,
        army_composition=army_composition,
        spell_composition=spell_composition,
        max_ticks=max_ticks,
    )
    obs, _ = env.reset(seed=seed, options={"profile": profile})
    assert env.sim is not None
    probe = EpisodeProbe(env.sim)
    rng = np.random.default_rng(seed + 31)
    steps = 0
    terminated = False
    truncated = False
    info: dict[str, Any] = {}
    while not env.sim.is_done:
        action = choose_action(model, obs, env.action_masks(), rng, deterministic)
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1

    sim = env.sim
    deploy_metrics = deployment_metrics(sim.deployment_cells)
    original_hp = max(1e-9, sim.original_army_hp)
    deployed_wb = army_composition.get("wall_breaker", 0) - sim.army_remaining_by_kind.get("wall_breaker", 0)
    alive_wb = sum(1 for troop in sim.troops if troop.spec.kind == "wall_breaker" and troop.alive)
    damage = probe.damage_by_source
    deaths = probe.deaths_by_source
    multi = probe.multi_hit_by_source
    splash_damage = sum(damage[kind] for kind in SPLASH_KINDS)
    splash_deaths = sum(deaths[kind] for kind in SPLASH_KINDS)
    return FailureEpisode(
        label=label,
        profile=profile,
        seed=seed,
        score=float(info.get("score", sim.score)),
        damage_pct=float(info.get("damage_pct", sim.damage_pct)),
        stars=int(info.get("stars", sim.stars)),
        steps=steps,
        ticks=sim.tick_count,
        terminated=int(bool(terminated)),
        truncated=int(bool(truncated)),
        deployments=int(deploy_metrics["deployments"]),
        deployment_unique_cells=int(deploy_metrics["deployment_unique_cells"]),
        deployment_top_cell_fraction=float(deploy_metrics["deployment_top_cell_fraction"]),
        deployment_top3_cell_fraction=float(deploy_metrics["deployment_top3_cell_fraction"]),
        deployment_quadrant_entropy=float(deploy_metrics["deployment_quadrant_entropy"]),
        active_nearest_distance_mean=float(np.mean(probe.active_nearest_samples)) if probe.active_nearest_samples else 0.0,
        active_close_pair_fraction_mean=float(np.mean(probe.active_close_samples)) if probe.active_close_samples else 0.0,
        active_quadrant_entropy_mean=float(np.mean(probe.active_quadrant_samples)) if probe.active_quadrant_samples else 0.0,
        troop_damage_taken_pct=sum(damage.values()) / original_hp,
        cannon_damage_pct=damage["cannon"] / original_hp,
        wizard_tower_damage_pct=damage["wizard_tower"] / original_hp,
        mortar_damage_pct=damage["mortar"] / original_hp,
        bomb_damage_pct=damage["bomb"] / original_hp,
        splash_damage_pct=splash_damage / original_hp,
        cannon_deaths=deaths["cannon"],
        wizard_tower_deaths=deaths["wizard_tower"],
        mortar_deaths=deaths["mortar"],
        bomb_deaths=deaths["bomb"],
        splash_deaths=splash_deaths,
        multi_hit_splash_events=sum(multi.values()),
        wizard_tower_multi_hit_events=multi["wizard_tower"],
        mortar_multi_hit_events=multi["mortar"],
        bomb_multi_hit_events=multi["bomb"],
        wall_breakers_deployed=int(deployed_wb),
        wall_breaker_explosions=probe.wall_breaker_explosions,
        wall_breakers_dead_without_explosion=max(0, int(deployed_wb) - probe.wall_breaker_explosions - int(alive_wb)),
        wall_segments_damaged_by_wb=probe.wall_segments_damaged_by_wb,
        wall_segments_destroyed_by_wb=probe.wall_segments_destroyed_by_wb,
        useful_rage_casts=int(info.get("useful_rage_casts", 0)),
        useful_freeze_casts=int(info.get("useful_freeze_casts", 0)),
        wasted_spell_casts=int(info.get("wasted_spell_casts", 0)),
        spells_remaining=int(info.get("spells_remaining", 0)),
        townhall_destroyed=int(info.get("townhall_destroyed", 0)),
        defenses_destroyed=int(info.get("defenses_destroyed", 0)),
    )


def worker(task: dict[str, Any]) -> list[dict[str, Any]]:
    deps = import_training_deps()
    try:
        deps.torch.set_num_threads(1)
        deps.torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    model_path = task["model_path"]
    model = None if model_path is None else deps.MaskablePPO.load(model_path, device=task["device"])
    rows = []
    for seed in task["seeds"]:
        rows.append(asdict(diagnose_episode(
            label=task["label"],
            model=model,
            profile=task["profile"],
            seed=seed,
            max_buildings=task["max_buildings"],
            army_composition=task["army_composition"],
            spell_composition=task["spell_composition"],
            max_ticks=task["max_ticks"],
            deterministic=task["deterministic"],
        )))
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    numeric_keys = [
        key for key, value in rows[0].items()
        if isinstance(value, (int, float)) and key != "seed"
    ]
    out: dict[str, Any] = {"episodes": len(rows)}
    for key in numeric_keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        out[f"{key}_mean"] = float(np.mean(values))
    out["p_stars_ge_1"] = float(np.mean([row["stars"] >= 1 for row in rows]))
    out["p_stars_ge_2"] = float(np.mean([row["stars"] >= 2 for row in rows]))
    out["p_damage_ge_50"] = float(np.mean([row["damage_pct"] >= 0.50 for row in rows]))
    out["p_damage_ge_90"] = float(np.mean([row["damage_pct"] >= 0.90 for row in rows]))
    out["p_townhall_destroyed"] = float(np.mean([row["townhall_destroyed"] >= 1 for row in rows]))
    return out


def print_summary(label: str, summary: dict[str, Any]) -> None:
    print(f"\n[{label}] episodes={summary['episodes']}")
    fields = [
        "score_mean", "stars_mean", "damage_pct_mean", "p_stars_ge_2", "p_townhall_destroyed",
        "troop_damage_taken_pct_mean", "splash_damage_pct_mean", "cannon_damage_pct_mean",
        "wizard_tower_damage_pct_mean", "mortar_damage_pct_mean", "bomb_damage_pct_mean",
        "splash_deaths_mean", "multi_hit_splash_events_mean",
        "deployment_unique_cells_mean", "deployment_top_cell_fraction_mean",
        "deployment_quadrant_entropy_mean", "active_nearest_distance_mean",
        "active_close_pair_fraction_mean", "active_quadrant_entropy_mean",
        "wall_breakers_deployed_mean", "wall_breaker_explosions_mean",
        "wall_breakers_dead_without_explosion_mean", "wall_segments_destroyed_by_wb_mean",
        "useful_rage_casts_mean", "useful_freeze_casts_mean", "spells_remaining_mean",
    ]
    for field in fields:
        print(f"{field}: {summary.get(field, 0.0):.4f}")


def chunked(items: list[int], chunks: int) -> list[list[int]]:
    return [items[i::chunks] for i in range(chunks) if items[i::chunks]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose hard-map failure modes with source-specific damage attribution.")
    parser.add_argument("--model", action="append", default=[], help="LABEL=path.zip. Use label=random with path=none for random.")
    parser.add_argument("--profile", default="hard")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=500_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--army-composition", default="barbarian=50,wall_breaker=15")
    parser.add_argument("--spell-composition", default="rage=4,freeze=4")
    parser.add_argument("--max-ticks", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/analysis/hard_failures"))
    args = parser.parse_args()

    specs: list[tuple[str, str | None]] = []
    for item in args.model:
        if "=" not in item:
            raise ValueError("--model must be LABEL=path.zip or LABEL=none")
        label, path = item.split("=", 1)
        specs.append((label, None if path.lower() == "none" else path))
    if not specs:
        specs.append(("random", None))

    profiles = profile_sequence(args.profile)
    if len(profiles) != 1:
        raise ValueError("diagnostic expects one profile")
    profile = profiles[0]
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    workers = max(1, min(args.workers, args.episodes))
    army_composition = parse_army_composition(args.army_composition)
    spell_composition = parse_spell_composition(args.spell_composition)
    max_buildings = max_preset_buildings()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for label, model_path in specs:
        tasks = [
            {
                "label": label,
                "model_path": model_path,
                "profile": profile,
                "seeds": seed_chunk,
                "max_buildings": max_buildings,
                "army_composition": army_composition,
                "spell_composition": spell_composition,
                "max_ticks": args.max_ticks,
                "deterministic": not args.stochastic,
                "device": args.device,
            }
            for seed_chunk in chunked(seeds, workers)
        ]
        rows: list[dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for part in pool.map(worker, tasks):
                rows.extend(part)
        rows.sort(key=lambda row: row["seed"])
        all_rows.extend(rows)
        print_summary(label, summarize(rows))

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = args.output_dir / f"hard_failures_{timestamp}.csv"
    json_path = args.output_dir / f"hard_failures_{timestamp}.json"
    summary = {label: summarize([row for row in all_rows if row["label"] == label]) for label, _ in specs}
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    with json_path.open("w") as fh:
        json.dump({"elapsed_seconds": time.perf_counter() - started, "summary": summary, "rows": all_rows}, fh, indent=2)
    print(f"\nwrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
