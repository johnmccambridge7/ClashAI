from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from coc_env.entities import DECISION_INTERVAL, GRID_SIZE
from coc_env.env import (
    CoCEnv,
    N_DEPLOY_ACTIONS,
    WAIT_ACTION,
    decode_deploy_action,
    decode_spell_action,
)
from scripts.random_baseline import summarize
from scripts.train_maskable_ppo import (
    DEFAULT_ARMY_COMPOSITION,
    DEFAULT_SPELL_COMPOSITION,
    format_composition,
    import_training_deps,
    parse_army_composition,
    parse_spell_composition,
    profile_sequence,
    resolve_model_path,
)


SPLASH_KINDS = {"wizard_tower", "mortar", "bomb"}


@dataclass
class EpisodeDiag:
    profile: str
    seed: int
    score: float
    damage_pct: float
    stars: int
    steps: int
    ticks: int
    deploys: int
    unique_deploy_cells: int
    top_deploy_cell_fraction: float
    top3_deploy_cell_fraction: float
    deploy_entropy_norm: float
    deploy_pair_distance_mean: float
    deploy_bbox_area_norm: float
    deployment_quadrant_entropy_norm: float
    active_pair_distance_mean: float
    active_nearest_distance_mean: float
    close_pair_fraction_mean: float
    splash_threat_cluster_risk_mean: float
    splash_event_ticks: int
    splash_damage_taken: float
    splash_deaths: int
    multi_troop_splash_ticks: int
    rage_casts: int
    rage_loose_useful_casts: int
    rage_strict_useful_casts: int
    rage_troops_covered_mean: float
    rage_attacking_troops_covered_mean: float
    freeze_casts: int
    freeze_loose_useful_casts: int
    freeze_strict_useful_casts: int
    freeze_defenses_covered_mean: float
    freeze_threatening_defenses_covered_mean: float


def choose_action(model: Any | None, obs: dict[str, np.ndarray], mask: np.ndarray, rng: np.random.Generator, deterministic: bool) -> int:
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        return WAIT_ACTION
    if model is None:
        return int(rng.choice(legal))
    action, _ = model.predict(obs, deterministic=deterministic, action_masks=mask)
    return int(np.asarray(action).item())


def troop_attacking_now(sim: Any, troop: Any) -> bool:
    if not troop.alive:
        return False
    target = sim._target_for(troop)
    if target is None:
        target_id = sim._pick_target_id(troop)
        target = sim._building_by_id(target_id) if target_id is not None else None
    if target is None:
        return False
    distance = target.distance_to(troop.x, troop.y)
    if troop.spec.explodes_on_wall and target.spec.kind == "wall":
        return distance <= troop.spec.explosion_trigger_range
    return distance <= troop.spec.attack_range + 0.7


def threatening_defense(sim: Any, defense: Any) -> bool:
    return sim._nearest_troop_in_range(defense) is not None


def quadrant(x: float, y: float) -> int:
    return int(x >= GRID_SIZE / 2.0) + 2 * int(y >= GRID_SIZE / 2.0)


def entropy_norm(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 1:
        return 0.0
    probs = [count / total for count in counts if count > 0]
    entropy = -sum(p * math.log(p) for p in probs)
    return float(entropy / max(1e-9, math.log(min(total, len(counts)))))


def pair_distances(points: list[tuple[float, float]]) -> list[float]:
    return [
        math.hypot(ax - bx, ay - by)
        for i, (ax, ay) in enumerate(points)
        for bx, by in points[i + 1 :]
    ]


def active_spread_metrics(sim: Any) -> tuple[float, float, float]:
    points = [(float(t.x), float(t.y)) for t in sim.active_troops]
    if len(points) < 2:
        return 0.0, 0.0, 0.0
    distances = pair_distances(points)
    nearest = []
    close = 0
    for i, (ax, ay) in enumerate(points):
        ds = [math.hypot(ax - bx, ay - by) for j, (bx, by) in enumerate(points) if i != j]
        nearest.append(min(ds))
        close += sum(1 for d in ds if d <= 1.5)
    pair_count = len(points) * (len(points) - 1)
    return float(np.mean(distances)), float(np.mean(nearest)), close / max(1, pair_count)


def splash_cluster_risk(sim: Any) -> float:
    troops = sim.active_troops
    if len(troops) < 2:
        return 0.0
    risk = 0.0
    for defense in sim.buildings:
        if not defense.alive or not defense.spec.is_defense or defense.spec.splash_radius <= 0.0:
            continue
        cx, cy = defense.center
        threatened = [
            troop
            for troop in troops
            if defense.spec.min_attack_range <= math.hypot(troop.x - cx, troop.y - cy) <= defense.spec.attack_range
        ]
        if len(threatened) < 2:
            continue
        for victim in threatened:
            risk += sum(
                1
                for other in troops
                if other.id != victim.id and math.hypot(other.x - victim.x, other.y - victim.y) <= defense.spec.splash_radius
            )
    return risk


def diagnose_episode(
    *,
    model: Any | None,
    profile: str,
    seed: int,
    max_buildings: int,
    army_composition: dict[str, int],
    spell_composition: dict[str, int],
    max_ticks: int,
    deterministic: bool,
) -> EpisodeDiag:
    env = CoCEnv(
        layout_profile=profile,
        max_buildings=max_buildings,
        army_composition=army_composition,
        spell_composition=spell_composition,
        max_ticks=max_ticks,
    )
    obs, _ = env.reset(seed=seed, options={"profile": profile})
    assert env.sim is not None
    sim = env.sim
    rng = np.random.default_rng(seed + 17)

    deployments: list[tuple[int, int, str]] = []
    spread_samples: list[tuple[float, float, float]] = []
    splash_risk_samples: list[float] = []
    splash_event_ticks = 0
    splash_damage_taken = 0.0
    splash_deaths = 0
    multi_troop_splash_ticks = 0

    rage_casts = rage_loose = rage_strict = 0
    rage_troops_covered: list[int] = []
    rage_attacking_covered: list[int] = []
    freeze_casts = freeze_loose = freeze_strict = 0
    freeze_defenses_covered: list[int] = []
    freeze_threatening_covered: list[int] = []

    steps = 0
    while not sim.is_done:
        mask = env.action_masks()
        action = choose_action(model, obs, mask, rng, deterministic)

        if 0 <= action < N_DEPLOY_ACTIONS and bool(mask[action]):
            troop_kind, x, y = decode_deploy_action(action)
            if env._can_deploy_cell(x, y) and sim.deploy(x + 0.5, y + 0.5, troop_kind=troop_kind):
                deployments.append((x, y, troop_kind))
        elif WAIT_ACTION < action < env.n_actions and bool(mask[action]):
            spell_kind, x, y = decode_spell_action(action)
            px, py = x + 0.5, y + 0.5
            radius = env.sim.spells_remaining_by_kind.get(spell_kind, 0)
            if radius > 0 and env._can_cast_spell_cell(x, y, spell_kind):
                spec = env.sim.active_spells[-1].spec if False and env.sim.active_spells else None
                spell_spec = __import__("coc_env.entities", fromlist=["SPELL_SPECS"]).SPELL_SPECS[spell_kind]
                if spell_kind == "rage":
                    troops = [t for t in sim.active_troops if math.hypot(t.x - px, t.y - py) <= spell_spec.radius]
                    attacking = [t for t in troops if troop_attacking_now(sim, t)]
                    rage_casts += 1
                    rage_loose += int(bool(troops))
                    rage_strict += int(bool(attacking))
                    rage_troops_covered.append(len(troops))
                    rage_attacking_covered.append(len(attacking))
                elif spell_kind == "freeze":
                    defenses = [
                        b for b in sim.buildings
                        if b.alive and b.spec.is_defense and math.hypot(b.center[0] - px, b.center[1] - py) <= spell_spec.radius
                    ]
                    threatening = [b for b in defenses if threatening_defense(sim, b)]
                    freeze_casts += 1
                    freeze_loose += int(bool(defenses))
                    freeze_strict += int(bool(threatening))
                    freeze_defenses_covered.append(len(defenses))
                    freeze_threatening_covered.append(len(threatening))
                sim.cast_spell(px, py, spell_kind=spell_kind)

        for _ in range(DECISION_INTERVAL):
            if sim.is_done:
                break
            spread_samples.append(active_spread_metrics(sim))
            splash_risk_samples.append(splash_cluster_risk(sim))
            hp_before = {t.id: max(0.0, t.hp) for t in sim.troops if t.alive}
            sim.tick()
            hp_after = {t.id: max(0.0, t.hp) for t in sim.troops}
            splash_events = [
                event for event in sim.visual_events
                if event.get("kind") == "impact"
                and str(event.get("source_kind")) in SPLASH_KINDS
                and float(event.get("radius") or 0.0) > 0.0
            ]
            if splash_events:
                damage = 0.0
                damaged = 0
                deaths = 0
                for tid, before in hp_before.items():
                    after = hp_after.get(tid, 0.0)
                    delta = max(0.0, before - after)
                    if delta > 0.0:
                        damage += delta
                        damaged += 1
                        deaths += int(after <= 0.0)
                splash_event_ticks += 1
                splash_damage_taken += damage
                splash_deaths += deaths
                multi_troop_splash_ticks += int(damaged > 1)
        obs = env._obs()
        steps += 1

    cell_counts: dict[tuple[int, int], int] = {}
    for x, y, _ in deployments:
        cell_counts[(x, y)] = cell_counts.get((x, y), 0) + 1
    counts = sorted(cell_counts.values(), reverse=True)
    points = [(x + 0.5, y + 0.5) for x, y, _ in deployments]
    q_counts = [0, 0, 0, 0]
    for x, y, _ in deployments:
        q_counts[quadrant(x + 0.5, y + 0.5)] += 1
    if points:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        bbox_area = (max(xs) - min(xs) + 1.0) * (max(ys) - min(ys) + 1.0) / (GRID_SIZE * GRID_SIZE)
        deploy_dists = pair_distances(points)
    else:
        bbox_area = 0.0
        deploy_dists = []
    spread = np.asarray(spread_samples, dtype=np.float64) if spread_samples else np.zeros((0, 3))

    return EpisodeDiag(
        profile=profile,
        seed=seed,
        score=float(sim.score),
        damage_pct=float(sim.damage_pct),
        stars=int(sim.stars),
        steps=steps,
        ticks=int(sim.tick_count),
        deploys=len(deployments),
        unique_deploy_cells=len(cell_counts),
        top_deploy_cell_fraction=counts[0] / len(deployments) if deployments and counts else 0.0,
        top3_deploy_cell_fraction=sum(counts[:3]) / len(deployments) if deployments else 0.0,
        deploy_entropy_norm=entropy_norm(counts),
        deploy_pair_distance_mean=float(np.mean(deploy_dists)) if deploy_dists else 0.0,
        deploy_bbox_area_norm=float(bbox_area),
        deployment_quadrant_entropy_norm=entropy_norm(q_counts),
        active_pair_distance_mean=float(spread[:, 0].mean()) if spread.size else 0.0,
        active_nearest_distance_mean=float(spread[:, 1].mean()) if spread.size else 0.0,
        close_pair_fraction_mean=float(spread[:, 2].mean()) if spread.size else 0.0,
        splash_threat_cluster_risk_mean=float(np.mean(splash_risk_samples)) if splash_risk_samples else 0.0,
        splash_event_ticks=splash_event_ticks,
        splash_damage_taken=float(splash_damage_taken),
        splash_deaths=splash_deaths,
        multi_troop_splash_ticks=multi_troop_splash_ticks,
        rage_casts=rage_casts,
        rage_loose_useful_casts=rage_loose,
        rage_strict_useful_casts=rage_strict,
        rage_troops_covered_mean=float(np.mean(rage_troops_covered)) if rage_troops_covered else 0.0,
        rage_attacking_troops_covered_mean=float(np.mean(rage_attacking_covered)) if rage_attacking_covered else 0.0,
        freeze_casts=freeze_casts,
        freeze_loose_useful_casts=freeze_loose,
        freeze_strict_useful_casts=freeze_strict,
        freeze_defenses_covered_mean=float(np.mean(freeze_defenses_covered)) if freeze_defenses_covered else 0.0,
        freeze_threatening_defenses_covered_mean=float(np.mean(freeze_threatening_covered)) if freeze_threatening_covered else 0.0,
    )


def worker(task: tuple[Any, ...]) -> list[dict[str, Any]]:
    checkpoint, profiles, seed_start, count, offset, max_buildings, army, spells, max_ticks, deterministic, device = task
    model = None
    if checkpoint is not None:
        deps = import_training_deps()
        try:
            deps.torch.set_num_threads(1)
        except Exception:
            pass
        model = deps.MaskablePPO.load(str(checkpoint), device=device)
    rows = []
    for i in range(count):
        seed = seed_start + offset + i
        profile = profiles[(offset + i) % len(profiles)]
        rows.append(asdict(diagnose_episode(
            model=model,
            profile=profile,
            seed=seed,
            max_buildings=max_buildings,
            army_composition=army,
            spell_composition=spells,
            max_ticks=max_ticks,
            deterministic=deterministic,
        )))
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        return {"episodes": 0}
    numeric_keys = [
        key for key, value in rows[0].items()
        if isinstance(value, (int, float)) and key != "seed"
    ]
    summary: dict[str, float | int] = {"episodes": len(rows)}
    for key in numeric_keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        summary[f"{key}_mean"] = float(values.mean())
    summary["p_two_star"] = float(np.mean([row["stars"] >= 2 for row in rows]))
    summary["p_damage_ge_90"] = float(np.mean([row["damage_pct"] >= 0.90 for row in rows]))
    summary["p_top_cell_gt_25"] = float(np.mean([row["top_deploy_cell_fraction"] > 0.25 for row in rows]))
    summary["p_top3_gt_50"] = float(np.mean([row["top3_deploy_cell_fraction"] > 0.50 for row in rows]))
    summary["strict_rage_useful_rate"] = (
        float(sum(row["rage_strict_useful_casts"] for row in rows) / max(1, sum(row["rage_casts"] for row in rows)))
    )
    summary["strict_freeze_useful_rate"] = (
        float(sum(row["freeze_strict_useful_casts"] for row in rows) / max(1, sum(row["freeze_casts"] for row in rows)))
    )
    summary["loose_to_strict_rage_gap"] = (
        float((sum(row["rage_loose_useful_casts"] for row in rows) - sum(row["rage_strict_useful_casts"] for row in rows)) / max(1, sum(row["rage_casts"] for row in rows)))
    )
    summary["loose_to_strict_freeze_gap"] = (
        float((sum(row["freeze_loose_useful_casts"] for row in rows) - sum(row["freeze_strict_useful_casts"] for row in rows)) / max(1, sum(row["freeze_casts"] for row in rows)))
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose policy deployment concentration, splash clumping, and spell timing.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Model zip or run dir. Omit for random legal actions.")
    parser.add_argument("--profile", default="hard")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=500000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--army-composition", default=format_composition(DEFAULT_ARMY_COMPOSITION))
    parser.add_argument("--spell-composition", default=format_composition(DEFAULT_SPELL_COMPOSITION))
    parser.add_argument("--max-ticks", type=int, default=900)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/diagnostics"))
    parser.add_argument("--name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    checkpoint = resolve_model_path(args.checkpoint) if args.checkpoint is not None else None
    profiles = profile_sequence(args.profile)
    army = parse_army_composition(args.army_composition)
    spells = parse_spell_composition(args.spell_composition)
    max_buildings = max(profile_sequence("hard") and [257])
    workers = min(args.workers, args.episodes)
    base = args.episodes // workers
    extra = args.episodes % workers
    tasks = []
    offset = 0
    for worker_id in range(workers):
        count = base + (1 if worker_id < extra else 0)
        tasks.append((checkpoint, profiles, args.seed_start, count, offset, max_buildings, army, spells, args.max_ticks, not args.stochastic, args.device))
        offset += count

    start = time.perf_counter()
    rows: list[dict[str, Any]] = []
    if workers == 1:
        for task in tasks:
            rows.extend(worker(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(worker, tasks):
                rows.extend(result)
    elapsed = time.perf_counter() - start
    rows.sort(key=lambda row: int(row["seed"]))
    summary = summarize_rows(rows)
    summary["elapsed_seconds"] = elapsed
    summary["episodes_per_min"] = len(rows) * 60.0 / max(1e-9, elapsed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"{'policy' if checkpoint else 'random'}_{'-'.join(profiles)}_n{len(rows)}_seed{args.seed_start}"
    output = args.output_dir / f"{name}.json"
    output.write_text(json.dumps({
        "config": {
            "checkpoint": str(checkpoint) if checkpoint is not None else None,
            "profiles": profiles,
            "episodes": args.episodes,
            "seed_start": args.seed_start,
            "army_composition": army,
            "spell_composition": spells,
            "max_ticks": args.max_ticks,
            "deterministic": not args.stochastic,
        },
        "summary": summary,
        "episodes": rows,
    }, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
