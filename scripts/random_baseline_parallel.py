from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time

from scripts.random_baseline import EpisodeResult, run_time_budget, summarize


def _worker(args: tuple[int, float, int, str]) -> tuple[int, list[EpisodeResult], float]:
    worker_id, seconds, seed, profile = args
    results, elapsed = run_time_budget(seconds=seconds, seed=seed, profile=profile)
    return worker_id, results, elapsed


def run_parallel(
    *,
    workers: int,
    seconds: float,
    seed: int,
    profile: str,
) -> tuple[list[EpisodeResult], float, list[tuple[int, int, float]]]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    if seconds <= 0.0:
        raise ValueError("seconds must be positive")

    tasks = [
        (worker_id, seconds, seed + worker_id * 1_000_003, profile)
        for worker_id in range(workers)
    ]
    start = time.perf_counter()
    all_results: list[EpisodeResult] = []
    worker_stats: list[tuple[int, int, float]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_worker, task) for task in tasks]
        for future in as_completed(futures):
            worker_id, results, elapsed = future.result()
            all_results.extend(results)
            worker_stats.append((worker_id, len(results), elapsed))
    return all_results, time.perf_counter() - start, sorted(worker_stats)


def main() -> None:
    default_workers = min(8, os.cpu_count() or 1)
    parser = argparse.ArgumentParser(
        description="Evaluate a uniformly random legal-action policy across multiple processes."
    )
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--workers", type=int, default=default_workers)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", default="hard", help="Preset profile name, or 'all'.")
    args = parser.parse_args()

    results, elapsed, worker_stats = run_parallel(
        workers=args.workers,
        seconds=args.seconds,
        seed=args.seed,
        profile=args.profile,
    )

    print(f"profile={args.profile} seed={args.seed} workers={args.workers}")
    print(summarize(results, elapsed))
    print(
        "workers "
        + " ".join(
            f"{worker_id}:{count}@{worker_elapsed:.1f}s"
            for worker_id, count, worker_elapsed in worker_stats
        )
    )


if __name__ == "__main__":
    main()
