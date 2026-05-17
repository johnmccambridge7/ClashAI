from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
RUN_ID = time.strftime("%Y%m%d_%H%M%S")
SUPERVISOR_DIR = REPO_ROOT / "runs" / "supervisor" / f"hard_followups_{RUN_ID}"
LOG_DIR = SUPERVISOR_DIR / "logs"

GPU0_CKPT = REPO_ROOT / "checkpoints" / "maskable_ppo" / "prod_medium_hard_v1_gpu0_seed45" / "final_model.zip"
GPU1_CKPT = REPO_ROOT / "checkpoints" / "maskable_ppo" / "prod_medium_hard_v1_gpu1_seed46" / "final_model.zip"


def _run(cmd: list[str], log_name: str, *, env: dict[str, str] | None = None) -> subprocess.Popen[bytes]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / log_name
    log = log_path.open("wb")
    print(f"starting {log_name}: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
    )
    return proc


def _wait(label: str, proc: subprocess.Popen[bytes]) -> None:
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"{label} exited with code {code}; see {LOG_DIR}")
    print(f"finished {label}", flush=True)


def _profile_metrics(eval_json: Path, profile: str) -> dict[str, float]:
    data = json.loads(eval_json.read_text(encoding="utf-8"))
    rows = [row for row in data["episodes"] if row["profile"] == profile]
    if not rows:
        raise RuntimeError(f"{eval_json} has no rows for profile {profile!r}")
    n = float(len(rows))
    return {
        "episodes": n,
        "score_mean": sum(float(row["score"]) for row in rows) / n,
        "damage_mean": sum(float(row["damage_pct"]) for row in rows) / n,
        "stars_mean": sum(float(row["stars"]) for row in rows) / n,
        "p_stars_ge_2": sum(float(row["stars"] >= 2) for row in rows) / n,
        "p_damage_ge_90": sum(float(row["damage_pct"] >= 0.9) for row in rows) / n,
    }


def _write_status(payload: dict[str, Any]) -> None:
    SUPERVISOR_DIR.mkdir(parents=True, exist_ok=True)
    path = SUPERVISOR_DIR / "status.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {path}", flush=True)


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    if not env.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY must be set for scheduled W&B runs")
    env.setdefault("WANDB_PROJECT", "ClashAI")
    env.setdefault("WANDB_ENTITY", "jmc314-lumenary-com")
    return env


def main() -> None:
    env = _base_env()

    eval_group = "prod_medium_hard_v1_holdout"
    eval_jobs = [
        (
            "gpu0",
            GPU0_CKPT,
            REPO_ROOT / "runs" / "eval" / "prod_medium_hard_v1_gpu0_seed45",
            "prod_medium_hard_v1_gpu0_holdout_600",
        ),
        (
            "gpu1",
            GPU1_CKPT,
            REPO_ROOT / "runs" / "eval" / "prod_medium_hard_v1_gpu1_seed46",
            "prod_medium_hard_v1_gpu1_holdout_600",
        ),
    ]

    eval_procs: list[tuple[str, subprocess.Popen[bytes]]] = []
    for label, checkpoint, output_dir, name in eval_jobs:
        cmd = [
            str(PYTHON),
            "scripts/evaluate_maskable_ppo.py",
            "--checkpoint",
            str(checkpoint),
            "--profile",
            "easy,medium,hard",
            "--episodes",
            "600",
            "--seed-start",
            "300000",
            "--workers",
            "48",
            "--device",
            "cpu",
            "--output-dir",
            str(output_dir),
            "--name",
            name,
            "--wandb",
            "--wandb-group",
            eval_group,
            "--wandb-tags",
            f"eval,holdout,prod_medium_hard_v1,{label}",
        ]
        eval_procs.append((label, _run(cmd, f"eval_{label}.log", env=env)))

    for label, proc in eval_procs:
        _wait(f"eval {label}", proc)

    summaries: dict[str, dict[str, float]] = {}
    checkpoints = {"gpu0": GPU0_CKPT, "gpu1": GPU1_CKPT}
    for label, _, output_dir, name in eval_jobs:
        summaries[label] = _profile_metrics(output_dir / f"{name}.json", "hard")

    best_label = max(summaries, key=lambda label: summaries[label]["score_mean"])
    best_checkpoint = checkpoints[best_label]
    _write_status({
        "phase": "launching_training",
        "best_label": best_label,
        "best_checkpoint": str(best_checkpoint),
        "hard_eval": summaries,
    })

    train_group = "hard_followups_v1"
    common = [
        str(PYTHON),
        "scripts/train_maskable_ppo.py",
        "--wandb",
        "--wandb-group",
        train_group,
        "--eval-profile",
        "easy,medium,hard",
        "--workers",
        "96",
        "--total-timesteps",
        "5000000",
        "--device",
        "cuda",
        "--n-steps",
        "512",
        "--batch-size",
        "8192",
        "--n-epochs",
        "4",
        "--target-kl",
        "0.03",
        "--checkpoint-freq",
        "500000",
        "--eval-freq",
        "0",
        "--status-interval",
        "5",
        "--monitor-dir",
        "runs/monitor",
        "--wandb-log-model",
    ]

    hard_focus_env = env.copy()
    hard_focus_env["CUDA_VISIBLE_DEVICES"] = "0"
    hard_v2_env = env.copy()
    hard_v2_env["CUDA_VISIBLE_DEVICES"] = "1"

    hard_focus_cmd = common + [
        "--run-name",
        "hard_focus_same_army_seed47",
        "--wandb-run-id",
        "hard_focus_same_army_seed47",
        "--wandb-tags",
        "train,hard_focus,same_army,warm_start",
        "--profile",
        "hard,hard,hard,medium",
        "--seed",
        "47",
        "--army-composition",
        "barbarian=40,wall_breaker=10",
        "--max-ticks",
        "720",
        "--learning-rate",
        "0.0002",
        "--ent-coef",
        "0.015",
        "--load",
        str(best_checkpoint),
    ]

    hard_v2_cmd = common + [
        "--run-name",
        "hard_v2_more_army_longer_seed48",
        "--wandb-run-id",
        "hard_v2_more_army_longer_seed48",
        "--wandb-tags",
        "train,hard_v2,more_army,longer_episode,from_scratch",
        "--profile",
        "hard,hard,hard,medium",
        "--seed",
        "48",
        "--army-composition",
        "barbarian=50,wall_breaker=15",
        "--max-ticks",
        "900",
        "--learning-rate",
        "0.0003",
        "--ent-coef",
        "0.015",
    ]

    train_procs = [
        ("hard_focus_same_army_seed47", _run(hard_focus_cmd, "train_hard_focus_same_army_seed47.log", env=hard_focus_env)),
        ("hard_v2_more_army_longer_seed48", _run(hard_v2_cmd, "train_hard_v2_more_army_longer_seed48.log", env=hard_v2_env)),
    ]
    _write_status({
        "phase": "training",
        "best_label": best_label,
        "best_checkpoint": str(best_checkpoint),
        "hard_eval": summaries,
        "runs": [name for name, _ in train_procs],
    })

    failures = []
    for label, proc in train_procs:
        code = proc.wait()
        print(f"finished train {label} with code {code}", flush=True)
        if code != 0:
            failures.append({"run": label, "code": code})

    _write_status({
        "phase": "completed" if not failures else "failed",
        "best_label": best_label,
        "best_checkpoint": str(best_checkpoint),
        "hard_eval": summaries,
        "failures": failures,
    })
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
