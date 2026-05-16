from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from coc_env.env import CoCEnv
from coc_env.generation import PRESET_LAYOUT_PROFILES
from scripts.random_baseline import EpisodeResult, summarize
from scripts.wandb_support import (
    finish_wandb_run,
    init_wandb_run,
    log_wandb_artifact,
    parse_wandb_tags,
)


DEFAULT_ARMY_COMPOSITION: dict[str, int] = {"barbarian": 40, "wall_breaker": 10}
DEFAULT_EVAL_SEED_START = 100_000


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(data), f, indent=2, sort_keys=True)
        f.write("\n")


def checkpoint_step(path: Path) -> int:
    stem = path.name
    if not stem.startswith("checkpoint_") or not stem.endswith("_steps.zip"):
        return -1
    try:
        return int(stem.removeprefix("checkpoint_").removesuffix("_steps.zip"))
    except ValueError:
        return -1


def resolve_model_path(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(path)
    for name in ("final_model.zip", "interrupted_model.zip", "best_model.zip"):
        candidate = path / name
        if candidate.exists():
            return candidate
    checkpoints = sorted(path.glob("checkpoint_*_steps.zip"), key=checkpoint_step)
    if checkpoints:
        return checkpoints[-1]
    raise FileNotFoundError(f"no model checkpoint found under {path}")


def set_thread_env_defaults(torch_threads: int) -> None:
    if torch_threads <= 0:
        return
    value = str(torch_threads)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, value)


def run_metadata() -> dict[str, Any]:
    return {
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
    }


@dataclass(frozen=True)
class TrainingDeps:
    MaskablePPO: Any
    MaskableMultiInputActorCriticPolicy: Any
    BaseCallback: Any
    CallbackList: Any
    CheckpointCallback: Any
    Monitor: Any
    DummyVecEnv: Any
    SubprocVecEnv: Any
    torch: Any


def profile_sequence(profile: str) -> list[str]:
    values = [part.strip() for part in profile.split(",") if part.strip()]
    if not values:
        raise ValueError("profile must not be empty")
    if values == ["all"]:
        return sorted(PRESET_LAYOUT_PROFILES)
    if "all" in values:
        raise ValueError("'all' cannot be combined with explicit profile names")
    unknown = [name for name in values if name not in PRESET_LAYOUT_PROFILES]
    if unknown:
        known = ", ".join(["all", *sorted(PRESET_LAYOUT_PROFILES)])
        raise ValueError(f"unknown profile(s) {unknown!r}; expected one of: {known}")
    return values


def parse_army_composition(value: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError("army composition entries must look like kind=count")
        kind, count_text = [piece.strip() for piece in part.split("=", 1)]
        if not kind:
            raise ValueError("army composition contains an empty troop kind")
        count = int(count_text)
        if count < 0:
            raise ValueError(f"{kind} count must be non-negative")
        out[kind] = count
    if not out:
        raise ValueError("army composition must contain at least one troop kind")
    return out


def max_preset_buildings() -> int:
    return max(profile.max_buildings for profile in PRESET_LAYOUT_PROFILES.values())


def linear_schedule(initial_value: float) -> Any:
    def schedule(progress_remaining: float) -> float:
        return float(progress_remaining) * initial_value

    return schedule


def gpu_snapshot() -> dict[str, float] | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2.0,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None

    rows: list[tuple[float, float, float]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
        except ValueError:
            continue
    if not rows:
        return None

    data = np.array(rows, dtype=np.float64)
    return {
        "gpu_count": float(len(rows)),
        "gpu_util_pct_mean": float(data[:, 0].mean()),
        "gpu_util_pct_max": float(data[:, 0].max()),
        "gpu_mem_used_gb_mean": float((data[:, 1] / 1024.0).mean()),
        "gpu_mem_total_gb_mean": float((data[:, 2] / 1024.0).mean()),
    }


def import_training_deps() -> TrainingDeps:
    try:
        import torch
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are not installed. Run: "
            ".venv/bin/pip install -e .[train,dev]"
        ) from exc

    return TrainingDeps(
        MaskablePPO=MaskablePPO,
        MaskableMultiInputActorCriticPolicy=MaskableMultiInputActorCriticPolicy,
        BaseCallback=BaseCallback,
        CallbackList=CallbackList,
        CheckpointCallback=CheckpointCallback,
        Monitor=Monitor,
        DummyVecEnv=DummyVecEnv,
        SubprocVecEnv=SubprocVecEnv,
        torch=torch,
    )


def make_env_factory(
    *,
    profile: str,
    seed: int,
    max_buildings: int,
    army_composition: dict[str, int],
    monitor_cls: Any,
    monitor_file: Path | None,
) -> Any:
    def _factory() -> Any:
        env = CoCEnv(
            layout_profile=profile,
            max_buildings=max_buildings,
            army_composition=army_composition,
        )
        env.reset(seed=seed)
        filename = str(monitor_file) if monitor_file is not None else None
        return monitor_cls(
            env,
            filename=filename,
            info_keywords=("profile", "damage_pct", "score", "stars", "ticks_elapsed"),
        )

    return _factory


def evaluate_model(
    model: Any,
    *,
    profiles: list[str],
    seed_start: int,
    episodes: int,
    max_buildings: int,
    army_composition: dict[str, int],
    deterministic: bool,
) -> tuple[list[EpisodeResult], float]:
    if episodes <= 0:
        return [], 0.0

    env = CoCEnv(
        layout_profile=profiles[0],
        max_buildings=max_buildings,
        army_composition=army_composition,
    )
    results: list[EpisodeResult] = []
    start = time.perf_counter()
    for episode in range(episodes):
        profile = profiles[episode % len(profiles)]
        seed = seed_start + episode
        obs, _ = env.reset(seed=seed, options={"profile": profile})
        terminated = truncated = False
        steps = 0
        info: dict[str, Any] = {
            "damage_pct": 0.0,
            "score": 0.0,
            "stars": 0,
            "ticks_elapsed": 0,
        }
        while not (terminated or truncated):
            action_masks = env.action_masks()
            action, _ = model.predict(
                obs,
                deterministic=deterministic,
                action_masks=action_masks,
            )
            action_int = int(np.asarray(action).item())
            obs, _, terminated, truncated, info = env.step(action_int)
            steps += 1
        results.append(EpisodeResult(
            profile=profile,
            seed=seed,
            damage_pct=float(info["damage_pct"]),
            score=float(info["score"]),
            stars=int(info["stars"]),
            steps=steps,
            ticks=int(info["ticks_elapsed"]),
            terminated=bool(terminated),
            truncated=bool(truncated),
        ))
    return results, time.perf_counter() - start


def mean_score(results: list[EpisodeResult]) -> float:
    if not results:
        return 0.0
    return float(np.mean([result.score for result in results]))


def result_metrics(results: list[EpisodeResult], elapsed: float) -> dict[str, float | int]:
    if not results:
        return {"episodes": 0, "elapsed_seconds": float(elapsed)}
    damage = np.array([result.damage_pct for result in results], dtype=np.float64)
    scores = np.array([result.score for result in results], dtype=np.float64)
    stars = np.array([result.stars for result in results], dtype=np.int64)
    steps = np.array([result.steps for result in results], dtype=np.float64)
    ticks = np.array([result.ticks for result in results], dtype=np.float64)
    return {
        "episodes": len(results),
        "elapsed_seconds": float(elapsed),
        "maps_per_min": float(len(results) / max(elapsed, 1e-9) * 60.0),
        "decisions_per_sec": float(steps.sum() / max(elapsed, 1e-9)),
        "sim_ticks_per_sec": float(ticks.sum() / max(elapsed, 1e-9)),
        "damage_mean": float(damage.mean()),
        "damage_median": float(np.median(damage)),
        "damage_min": float(damage.min()),
        "damage_max": float(damage.max()),
        "damage_std": float(damage.std()),
        "score_mean": float(scores.mean()),
        "score_median": float(np.median(scores)),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "stars_mean": float(stars.mean()),
        "stars_0": int((stars == 0).sum()),
        "stars_1": int((stars == 1).sum()),
        "stars_2": int((stars == 2).sum()),
        "stars_3": int((stars == 3).sum()),
        "p_stars_ge_2": float((stars >= 2).mean()),
        "p_damage_ge_25": float((damage >= 0.25).mean()),
        "p_damage_ge_50": float((damage >= 0.50).mean()),
        "p_damage_ge_75": float((damage >= 0.75).mean()),
        "p_damage_ge_90": float((damage >= 0.90).mean()),
        "p_damage_eq_100": float((damage >= 1.0).mean()),
        "steps_mean": float(steps.mean()),
        "ticks_mean": float(ticks.mean()),
        "terminated": int(sum(result.terminated for result in results)),
        "truncated": int(sum(result.truncated for result in results)),
    }


def _window_mean(records: deque[dict[str, float]], key: str) -> float:
    values = [record[key] for record in records if key in record]
    return float(np.mean(values)) if values else 0.0


def _window_rate_at_least(records: deque[dict[str, float]], key: str, threshold: float) -> float:
    values = [record[key] for record in records if key in record]
    return float(np.mean(np.asarray(values, dtype=np.float64) >= threshold)) if values else 0.0


def _profile_metric_name(profile: str) -> str:
    return profile.replace("/", "_").replace(" ", "_")


def make_throughput_callback(base_callback_cls: Any, *, wandb_run: Any | None = None) -> type:
    class ThroughputCallback(base_callback_cls):  # type: ignore[misc, valid-type]
        def __init__(self, *, interval_seconds: float, window_size: int = 200, verbose: int = 0) -> None:
            super().__init__(verbose=verbose)
            self.interval_seconds = interval_seconds
            self.window_size = window_size
            self._last_time = 0.0
            self._last_timesteps = 0
            self._last_episodes = 0
            self._episodes = 0
            self._profile_episodes: dict[str, deque[dict[str, float]]] = {}

        def _on_training_start(self) -> None:
            now = time.perf_counter()
            self._last_time = now
            self._last_timesteps = int(self.num_timesteps)
            self._last_episodes = 0

        def _on_step(self) -> bool:
            dones = self.locals.get("dones")
            if dones is not None:
                self._episodes += int(np.asarray(dones, dtype=bool).sum())
            self._record_completed_episodes()

            now = time.perf_counter()
            elapsed = now - self._last_time
            if elapsed < self.interval_seconds:
                return True

            step_delta = int(self.num_timesteps) - self._last_timesteps
            episode_delta = self._episodes - self._last_episodes
            env_steps_per_sec = step_delta / max(elapsed, 1e-9)
            episodes_per_sec = episode_delta / max(elapsed, 1e-9)
            self.logger.record("throughput/env_steps_per_sec", env_steps_per_sec)
            self.logger.record("throughput/episodes_per_sec", episodes_per_sec)
            wandb_payload: dict[str, float | int] = {
                "train/total_timesteps": int(self.num_timesteps),
                "throughput/env_steps_per_sec": float(env_steps_per_sec),
                "throughput/episodes_per_sec": float(episodes_per_sec),
            }

            parts = [
                f"timesteps={self.num_timesteps}",
                f"env_steps/sec={env_steps_per_sec:.1f}",
                f"episodes/sec={episodes_per_sec:.2f}",
            ]
            gpu = gpu_snapshot()
            if gpu is not None:
                for key, value in gpu.items():
                    self.logger.record(f"gpu/{key}", value)
                    wandb_payload[f"gpu/{key}"] = float(value)
                parts.append(
                    f"gpu_util_mean={gpu['gpu_util_pct_mean']:.0f}% "
                    f"gpu_util_max={gpu['gpu_util_pct_max']:.0f}%"
                )
                parts.append(
                    f"gpu_mem_mean={gpu['gpu_mem_used_gb_mean']:.1f}/"
                    f"{gpu['gpu_mem_total_gb_mean']:.1f}GB"
                )
            self._log_profile_windows(wandb_payload)
            if wandb_run is not None:
                wandb_run.log(wandb_payload)
            print(" | ".join(parts), flush=True)

            self._last_time = now
            self._last_timesteps = int(self.num_timesteps)
            self._last_episodes = self._episodes
            return True

        def _record_completed_episodes(self) -> None:
            infos = self.locals.get("infos") or []
            for info in infos:
                if not isinstance(info, dict):
                    continue
                episode = info.get("episode")
                if not isinstance(episode, dict):
                    continue
                profile = str(episode.get("profile") or info.get("profile") or "unknown")
                records = self._profile_episodes.setdefault(profile, deque(maxlen=self.window_size))
                records.append({
                    "reward": float(episode.get("r", 0.0)),
                    "length": float(episode.get("l", 0.0)),
                    "damage_pct": float(episode.get("damage_pct", info.get("damage_pct", 0.0))),
                    "score": float(episode.get("score", info.get("score", 0.0))),
                    "stars": float(episode.get("stars", info.get("stars", 0))),
                    "ticks_elapsed": float(episode.get("ticks_elapsed", info.get("ticks_elapsed", 0))),
                    "truncated": float(bool(info.get("TimeLimit.truncated") or info.get("is_truncated"))),
                })

        def _log_profile_windows(self, wandb_payload: dict[str, float | int]) -> None:
            for profile, records in sorted(self._profile_episodes.items()):
                if not records:
                    continue
                prefix = f"profile/{_profile_metric_name(profile)}"
                metrics = {
                    "episodes_window": float(len(records)),
                    "ep_rew_mean": _window_mean(records, "reward"),
                    "ep_len_mean": _window_mean(records, "length"),
                    "score_mean": _window_mean(records, "score"),
                    "damage_mean": _window_mean(records, "damage_pct"),
                    "stars_mean": _window_mean(records, "stars"),
                    "ticks_mean": _window_mean(records, "ticks_elapsed"),
                    "p_stars_ge_2": _window_rate_at_least(records, "stars", 2.0),
                    "p_damage_ge_90": _window_rate_at_least(records, "damage_pct", 0.90),
                    "truncated_rate": _window_rate_at_least(records, "truncated", 1.0),
                }
                for key, value in metrics.items():
                    metric_name = f"{prefix}/{key}"
                    self.logger.record(metric_name, value)
                    wandb_payload[metric_name] = float(value)

    return ThroughputCallback


def make_holdout_eval_callback(base_callback_cls: Any) -> type:
    class HoldoutEvalCallback(base_callback_cls):  # type: ignore[misc, valid-type]
        def __init__(
            self,
            *,
            eval_freq: int,
            profiles: list[str],
            seed_start: int,
            episodes: int,
            max_buildings: int,
            army_composition: dict[str, int],
            deterministic: bool,
            best_model_path: Path | None,
            wandb_run: Any | None = None,
            verbose: int = 0,
        ) -> None:
            super().__init__(verbose=verbose)
            self.eval_freq = eval_freq
            self.profiles = profiles
            self.seed_start = seed_start
            self.episodes = episodes
            self.max_buildings = max_buildings
            self.army_composition = army_composition
            self.deterministic = deterministic
            self.best_model_path = best_model_path
            self.wandb_run = wandb_run
            self.best_score = float("-inf")
            self._last_eval_timesteps = 0

        def _on_step(self) -> bool:
            if self.eval_freq <= 0:
                return True
            if int(self.num_timesteps) - self._last_eval_timesteps < self.eval_freq:
                return True
            self._last_eval_timesteps = int(self.num_timesteps)

            results, elapsed = evaluate_model(
                self.model,
                profiles=self.profiles,
                seed_start=self.seed_start,
                episodes=self.episodes,
                max_buildings=self.max_buildings,
                army_composition=self.army_composition,
                deterministic=self.deterministic,
            )
            if not results:
                return True

            damage = np.array([result.damage_pct for result in results], dtype=np.float64)
            scores = np.array([result.score for result in results], dtype=np.float64)
            stars = np.array([result.stars for result in results], dtype=np.int64)
            steps = np.array([result.steps for result in results], dtype=np.float64)
            ticks = np.array([result.ticks for result in results], dtype=np.float64)

            self.logger.record("eval/episodes", len(results))
            self.logger.record("eval/elapsed_seconds", elapsed)
            self.logger.record("eval/damage_mean", float(damage.mean()))
            self.logger.record("eval/damage_median", float(np.median(damage)))
            self.logger.record("eval/score_mean", float(scores.mean()))
            self.logger.record("eval/score_median", float(np.median(scores)))
            self.logger.record("eval/stars_mean", float(stars.mean()))
            self.logger.record("eval/steps_mean", float(steps.mean()))
            self.logger.record("eval/ticks_mean", float(ticks.mean()))
            self.logger.record("eval/p_damage_ge_50", float((damage >= 0.50).mean()))
            self.logger.record("eval/p_damage_ge_90", float((damage >= 0.90).mean()))
            self.logger.record("eval/p_stars_ge_2", float((stars >= 2).mean()))
            for star in range(4):
                self.logger.record(f"eval/stars_{star}", float((stars == star).mean()))
            if self.wandb_run is not None:
                payload: dict[str, float | int] = {"train/total_timesteps": int(self.num_timesteps)}
                for key, value in result_metrics(results, elapsed).items():
                    payload[f"eval/{key}"] = value
                for profile in sorted({result.profile for result in results}):
                    profile_results = [result for result in results if result.profile == profile]
                    prefix = f"eval_profile/{_profile_metric_name(profile)}"
                    for key, value in result_metrics(profile_results, elapsed).items():
                        payload[f"{prefix}/{key}"] = value
                self.wandb_run.log(payload)

            score = mean_score(results)
            if self.best_model_path is not None and score > self.best_score:
                self.best_score = score
                self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
                self.model.save(str(self.best_model_path))
                if self.wandb_run is not None:
                    self.wandb_run.summary["best_eval_score"] = score

            print(
                f"\n[eval timesteps={self.num_timesteps} profiles={','.join(self.profiles)} "
                f"seed_start={self.seed_start} episodes={self.episodes}]",
                flush=True,
            )
            print(summarize(results, elapsed), flush=True)
            if self.best_model_path is not None:
                print(f"best_eval_score={self.best_score:.3f}", flush=True)
            return True

    return HoldoutEvalCallback


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MaskablePPO on the ClashAI Gymnasium environment.")
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--profile", default="hard", help="Preset profile, comma-list, or 'all'.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--army-composition", default="barbarian=40,wall_breaker=10")

    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--linear-lr", action="store_true")
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)

    parser.add_argument("--vec-env", choices=("auto", "subproc", "dummy"), default="auto")
    parser.add_argument("--start-method", default="forkserver")
    parser.add_argument("--log-dir", type=Path, default=Path("runs/maskable_ppo"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--monitor-dir", type=Path, default=None)
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints/maskable_ppo"))
    parser.add_argument("--checkpoint-freq", type=int, default=250_000)
    parser.add_argument("--load", type=Path, default=None, help="Model zip or run directory to load.")
    parser.add_argument("--resume", action="store_true", help="Resume from --save-dir/--run-name by resolving final/interrupted/best/latest checkpoint.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config, write manifest, and exit before building envs.")
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--status-interval", type=float, default=30.0)

    parser.add_argument("--wandb", action="store_true", help="Stream training metrics and artifacts to W&B.")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "clashai-rl"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-tags", default=None, help="Comma-separated W&B tags.")
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--wandb-run-id", default=None)
    parser.add_argument("--wandb-log-code", action="store_true")
    parser.add_argument("--wandb-log-model", action="store_true")
    parser.add_argument("--no-wandb-sync-tensorboard", dest="wandb_sync_tensorboard", action="store_false")
    parser.set_defaults(wandb_sync_tensorboard=True)

    parser.add_argument("--eval-profile", default=None, help="Default: same as --profile.")
    parser.add_argument("--eval-freq", type=int, default=0, help="Set >0 for serial in-loop eval; production runs should usually evaluate out of process.")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed-start", type=int, default=DEFAULT_EVAL_SEED_START)
    parser.add_argument("--stochastic-eval", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    set_thread_env_defaults(args.torch_threads)
    deps = import_training_deps()

    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.total_timesteps <= 0:
        raise ValueError("total-timesteps must be positive")
    if args.resume and not args.run_name:
        raise ValueError("--resume requires --run-name so the existing run directory can be resolved")

    if args.torch_threads > 0:
        deps.torch.set_num_threads(args.torch_threads)

    train_profiles = profile_sequence(args.profile)
    eval_profiles = profile_sequence(args.eval_profile or args.profile)
    army_composition = parse_army_composition(args.army_composition)
    max_buildings = max_preset_buildings()

    run_name = args.run_name or (
        f"{'-'.join(train_profiles)}_w{args.workers}_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    tensorboard_log = args.log_dir / run_name
    save_dir = args.save_dir / run_name
    tensorboard_log.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    load_path = args.load
    if args.resume:
        if load_path is not None:
            raise ValueError("--resume and --load cannot be combined")
        load_path = resolve_model_path(save_dir)

    run_config = {
        "run_name": run_name,
        "train_profiles": train_profiles,
        "eval_profiles": eval_profiles,
        "army_composition": army_composition,
        "max_buildings": max_buildings,
        "total_timesteps": args.total_timesteps,
        "workers": args.workers,
        "seed": args.seed,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "learning_rate": args.learning_rate,
        "linear_lr": args.linear_lr,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
        "device": args.device,
        "torch_threads": args.torch_threads,
        "vec_env": args.vec_env,
        "start_method": args.start_method,
        "checkpoint_freq": args.checkpoint_freq,
        "eval_freq": args.eval_freq,
        "eval_episodes": args.eval_episodes,
        "eval_seed_start": args.eval_seed_start,
        "stochastic_eval": args.stochastic_eval,
        "load_path": str(load_path) if load_path is not None else None,
        "paths": {
            "tensorboard_log": tensorboard_log,
            "save_dir": save_dir,
            "monitor_dir": args.monitor_dir / run_name if args.monitor_dir is not None else None,
        },
        "wandb": {
            "enabled": args.wandb,
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "group": args.wandb_group,
            "tags": parse_wandb_tags(args.wandb_tags),
            "mode": args.wandb_mode,
            "run_id": args.wandb_run_id or run_name,
            "sync_tensorboard": args.wandb_sync_tensorboard,
            "log_code": args.wandb_log_code,
            "log_model": args.wandb_log_model,
        },
        "metadata": run_metadata(),
    }
    write_json(save_dir / "run_config.json", run_config)
    write_json(tensorboard_log / "run_config.json", run_config)
    if args.dry_run:
        print(json.dumps(_jsonable(run_config), indent=2, sort_keys=True), flush=True)
        return

    wandb_run = init_wandb_run(
        enabled=args.wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=_jsonable(run_config),
        group=args.wandb_group,
        job_type="train",
        tags=parse_wandb_tags(args.wandb_tags),
        mode=args.wandb_mode,
        run_id=args.wandb_run_id or run_name,
        sync_tensorboard=args.wandb_sync_tensorboard,
        save_code=args.wandb_log_code,
    )
    log_wandb_artifact(
        wandb_run,
        save_dir / "run_config.json",
        name=f"{run_name}-run-config",
        artifact_type="run_config",
    )

    monitor_dir = args.monitor_dir
    if monitor_dir is not None:
        monitor_dir = monitor_dir / run_name
        monitor_dir.mkdir(parents=True, exist_ok=True)

    env_fns = []
    for rank in range(args.workers):
        profile = train_profiles[rank % len(train_profiles)]
        seed = args.seed + rank * 100_003
        monitor_file = monitor_dir / f"worker_{rank}" if monitor_dir is not None else None
        env_fns.append(make_env_factory(
            profile=profile,
            seed=seed,
            max_buildings=max_buildings,
            army_composition=army_composition,
            monitor_cls=deps.Monitor,
            monitor_file=monitor_file,
        ))

    vec_kind = args.vec_env
    if vec_kind == "auto":
        vec_kind = "dummy" if args.workers == 1 else "subproc"
    if vec_kind == "dummy":
        env = deps.DummyVecEnv(env_fns)
    else:
        env = deps.SubprocVecEnv(env_fns, start_method=args.start_method)

    lr = linear_schedule(args.learning_rate) if args.linear_lr else args.learning_rate
    if load_path is not None:
        model = deps.MaskablePPO.load(
            str(load_path),
            env=env,
            device=args.device,
            tensorboard_log=str(args.log_dir),
        )
        reset_num_timesteps = False
    else:
        model = deps.MaskablePPO(
            deps.MaskableMultiInputActorCriticPolicy,
            env,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            learning_rate=lr,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            tensorboard_log=str(args.log_dir),
            device=args.device,
            verbose=1,
        )
        reset_num_timesteps = True

    callbacks = []
    ThroughputCallback = make_throughput_callback(deps.BaseCallback, wandb_run=wandb_run)
    if args.status_interval > 0:
        callbacks.append(ThroughputCallback(interval_seconds=args.status_interval))

    if args.checkpoint_freq > 0:
        callbacks.append(deps.CheckpointCallback(
            save_freq=max(args.checkpoint_freq // args.workers, 1),
            save_path=str(save_dir),
            name_prefix="checkpoint",
            save_replay_buffer=False,
            save_vecnormalize=False,
        ))

    HoldoutEvalCallback = make_holdout_eval_callback(deps.BaseCallback)
    if args.eval_freq > 0 and args.eval_episodes > 0:
        callbacks.append(HoldoutEvalCallback(
            eval_freq=args.eval_freq,
            profiles=eval_profiles,
            seed_start=args.eval_seed_start,
            episodes=args.eval_episodes,
            max_buildings=max_buildings,
            army_composition=army_composition,
            deterministic=not args.stochastic_eval,
            best_model_path=save_dir / "best_model",
            wandb_run=wandb_run,
        ))

    callback = deps.CallbackList(callbacks) if callbacks else None
    print(
        "training "
        f"profiles={','.join(train_profiles)} eval_profiles={','.join(eval_profiles)} "
        f"workers={args.workers} device={args.device} timesteps={args.total_timesteps} "
        f"run={run_name}",
        flush=True,
    )
    train_start = time.perf_counter()
    status: dict[str, Any] = {"status": "running", "run_name": run_name, "started_at_unix": time.time()}
    write_json(save_dir / "run_status.json", status)
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            tb_log_name=run_name,
            reset_num_timesteps=reset_num_timesteps,
            log_interval=args.log_interval,
            progress_bar=args.progress_bar,
        )
        final_path = save_dir / "final_model"
        model.save(str(final_path))
        status.update({"status": "completed", "final_model": final_path.with_suffix(".zip")})
        if args.wandb_log_model:
            log_wandb_artifact(
                wandb_run,
                final_path.with_suffix(".zip"),
                name=f"{run_name}-final-model",
                artifact_type="model",
                metadata={"run_name": run_name, "status": "completed"},
            )
        print(f"saved final model: {final_path}", flush=True)
    except KeyboardInterrupt:
        interrupted_path = save_dir / "interrupted_model"
        model.save(str(interrupted_path))
        status.update({"status": "interrupted", "interrupted_model": interrupted_path.with_suffix(".zip")})
        if args.wandb_log_model:
            log_wandb_artifact(
                wandb_run,
                interrupted_path.with_suffix(".zip"),
                name=f"{run_name}-interrupted-model",
                artifact_type="model",
                metadata={"run_name": run_name, "status": "interrupted"},
            )
        print(f"saved interrupted model: {interrupted_path}", flush=True)
        raise
    except Exception as exc:
        status.update({"status": "failed", "error": repr(exc)})
        raise
    finally:
        status.update({
            "finished_at_unix": time.time(),
            "elapsed_seconds": time.perf_counter() - train_start,
            "num_timesteps": int(getattr(model, "num_timesteps", 0)),
        })
        write_json(save_dir / "run_status.json", status)
        log_wandb_artifact(
            wandb_run,
            save_dir / "run_status.json",
            name=f"{run_name}-run-status",
            artifact_type="run_status",
        )
        finish_wandb_run(wandb_run, status=str(status.get("status", "")))
        env.close()


if __name__ == "__main__":
    main()
