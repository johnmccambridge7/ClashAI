from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from scripts.random_baseline import EpisodeResult, summarize
from scripts.train_maskable_ppo import (
    DEFAULT_ARMY_COMPOSITION,
    DEFAULT_SPELL_COMPOSITION,
    DEFAULT_EVAL_SEED_START,
    evaluate_model,
    format_composition,
    import_training_deps,
    max_preset_buildings,
    parse_army_composition,
    parse_spell_composition,
    profile_sequence,
    resolve_model_path,
    result_metrics,
    write_json,
)
from scripts.wandb_support import (
    DEFAULT_WANDB_ENTITY,
    DEFAULT_WANDB_PROJECT,
    finish_wandb_run,
    init_wandb_run,
    log_wandb_artifact,
    log_wandb_table,
    parse_wandb_tags,
)


def summary_dict(results: list[EpisodeResult], elapsed: float) -> dict[str, Any]:
    return result_metrics(results, elapsed)


def write_results(
    output_dir: Path,
    name: str,
    results: list[EpisodeResult],
    elapsed: float,
    config: dict[str, Any],
) -> tuple[Path, Path | None, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [asdict(result) for result in results]
    summary = summary_dict(results, elapsed)
    json_path = output_dir / f"{name}.json"
    csv_path = output_dir / f"{name}.csv" if records else None
    write_json(json_path, {"config": config, "summary": summary, "episodes": records})
    if records:
        assert csv_path is not None
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    return json_path, csv_path, summary


def _rotated_profiles(profiles: list[str], offset: int) -> list[str]:
    if not profiles:
        return profiles
    i = offset % len(profiles)
    return profiles[i:] + profiles[:i]


def _eval_worker(task: tuple[str, list[str], int, int, int, dict[str, int], dict[str, int], int, bool, str, int]) -> tuple[int, list[EpisodeResult], float]:
    checkpoint, profiles, seed_start, episodes, max_buildings, army_composition, spell_composition, max_ticks, deterministic, device, episode_offset = task
    deps = import_training_deps()
    model = deps.MaskablePPO.load(checkpoint, device=device)
    worker_profiles = _rotated_profiles(profiles, episode_offset)
    results, elapsed = evaluate_model(
        model,
        profiles=worker_profiles,
        seed_start=seed_start,
        episodes=episodes,
        max_buildings=max_buildings,
        army_composition=army_composition,
        spell_composition=spell_composition,
        max_ticks=max_ticks,
        deterministic=deterministic,
    )
    return episode_offset, results, elapsed


def evaluate_parallel(
    *,
    checkpoint: Path,
    profiles: list[str],
    seed_start: int,
    episodes: int,
    max_buildings: int,
    army_composition: dict[str, int],
    spell_composition: dict[str, int],
    max_ticks: int,
    deterministic: bool,
    device: str,
    workers: int,
) -> tuple[list[EpisodeResult], float]:
    if workers <= 1 or episodes <= 1:
        deps = import_training_deps()
        model = deps.MaskablePPO.load(str(checkpoint), device=device)
        return evaluate_model(
            model,
            profiles=profiles,
            seed_start=seed_start,
            episodes=episodes,
            max_buildings=max_buildings,
            army_composition=army_composition,
            spell_composition=spell_composition,
            max_ticks=max_ticks,
            deterministic=deterministic,
        )

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
            str(checkpoint),
            profiles,
            seed_start + offset,
            count,
            max_buildings,
            army_composition,
            spell_composition,
            max_ticks,
            deterministic,
            device,
            offset,
        ))
        offset += count

    start = time.perf_counter()
    chunks: list[tuple[int, list[EpisodeResult], float]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_eval_worker, task) for task in tasks]
        for future in as_completed(futures):
            chunks.append(future.result())
    results: list[EpisodeResult] = []
    for _, chunk, _ in sorted(chunks, key=lambda item: item[0]):
        results.extend(chunk)
    return results, time.perf_counter() - start


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a MaskablePPO ClashAI checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model zip or run directory.")
    parser.add_argument("--profile", default="medium", help="Preset profile, comma-list, or 'all'.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_EVAL_SEED_START)
    parser.add_argument("--army-composition", default=format_composition(DEFAULT_ARMY_COMPOSITION))
    parser.add_argument("--spell-composition", default=format_composition(DEFAULT_SPELL_COMPOSITION))
    parser.add_argument("--max-ticks", type=int, default=720)
    parser.add_argument("--device", default="cpu", help="Use cpu for parallel eval unless there is a reason to share cuda.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/eval"))
    parser.add_argument("--name", default=None)
    parser.add_argument("--wandb", action="store_true", help="Log evaluation summary, table, and artifacts to W&B.")
    parser.add_argument("--wandb-project", default=None, help="Default: WANDB_PROJECT or ClashAI.")
    parser.add_argument("--wandb-entity", default=None, help="Default: WANDB_ENTITY or the repo W&B entity.")
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-tags", default=None, help="Comma-separated W&B tags.")
    parser.add_argument("--wandb-mode", default=None, help="Default: WANDB_MODE or online.")
    parser.add_argument("--wandb-run-id", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.max_ticks <= 0:
        raise ValueError("max-ticks must be positive")

    checkpoint = resolve_model_path(args.checkpoint)
    profiles = profile_sequence(args.profile)
    army_composition = parse_army_composition(args.army_composition)
    spell_composition = parse_spell_composition(args.spell_composition)
    max_buildings = max_preset_buildings()
    name = args.name or f"{'-'.join(profiles)}_n{args.episodes}_seed{args.seed_start}"

    config = {
        "checkpoint": checkpoint,
        "profiles": profiles,
        "episodes": args.episodes,
        "seed_start": args.seed_start,
        "army_composition": army_composition,
        "spell_composition": spell_composition,
        "max_ticks": args.max_ticks,
        "max_buildings": max_buildings,
        "deterministic": not args.stochastic,
        "device": args.device,
        "workers": args.workers,
        "wandb": {
            "enabled": args.wandb,
            "project": args.wandb_project or os.environ.get("WANDB_PROJECT", DEFAULT_WANDB_PROJECT),
            "entity": args.wandb_entity or os.environ.get("WANDB_ENTITY", DEFAULT_WANDB_ENTITY),
            "group": args.wandb_group,
            "tags": parse_wandb_tags(args.wandb_tags),
            "mode": args.wandb_mode or os.environ.get("WANDB_MODE", "online"),
            "run_id": args.wandb_run_id,
        },
    }
    print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in config.items()}, indent=2), flush=True)
    wandb_run = None
    wandb_status = "failed"
    try:
        results, elapsed = evaluate_parallel(
            checkpoint=checkpoint,
            profiles=profiles,
            seed_start=args.seed_start,
            episodes=args.episodes,
            max_buildings=max_buildings,
            army_composition=army_composition,
            spell_composition=spell_composition,
            max_ticks=args.max_ticks,
            deterministic=not args.stochastic,
            device=args.device,
            workers=args.workers,
        )
        print(summarize(results, elapsed), flush=True)
        json_path, csv_path, summary = write_results(args.output_dir, name, results, elapsed, config)
        print(f"wrote: {json_path}", flush=True)

        if args.wandb:
            # Initialize W&B only after ProcessPool evaluation completes. W&B
            # starts background threads/processes, and forking workers after
            # that can leave parallel eval workers idle.
            wandb_run = init_wandb_run(
                enabled=True,
                project=config["wandb"]["project"],
                entity=config["wandb"]["entity"],
                name=f"eval-{name}",
                config={k: str(v) if isinstance(v, Path) else v for k, v in config.items()},
                group=args.wandb_group,
                job_type="eval",
                tags=parse_wandb_tags(args.wandb_tags),
                mode=config["wandb"]["mode"],
                run_id=args.wandb_run_id,
                step_metric="eval/episodes",
            )
            records = [asdict(result) for result in results]
            payload: dict[str, Any] = {f"eval/{key}": value for key, value in summary.items()}
            for profile in sorted({result.profile for result in results}):
                profile_results = [result for result in results if result.profile == profile]
                profile_elapsed = elapsed * len(profile_results) / max(1, len(results))
                for key, value in summary_dict(profile_results, profile_elapsed).items():
                    payload[f"eval_profile/{profile}/{key}"] = value
            wandb_run.log(payload)
            log_wandb_table(wandb_run, "eval/episodes_table", records)
            log_wandb_artifact(
                wandb_run,
                json_path,
                name=f"{name}-eval-json",
                artifact_type="eval",
                metadata={"profiles": profiles, "episodes": args.episodes},
            )
            if csv_path is not None:
                log_wandb_artifact(
                    wandb_run,
                    csv_path,
                    name=f"{name}-eval-csv",
                    artifact_type="eval",
                    metadata={"profiles": profiles, "episodes": args.episodes},
                )
        wandb_status = "completed"
    finally:
        finish_wandb_run(wandb_run, status=wandb_status)


if __name__ == "__main__":
    main()
