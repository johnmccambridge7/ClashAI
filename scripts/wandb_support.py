from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any


def parse_wandb_tags(value: str | None) -> list[str]:
    if value is None:
        return []
    return [tag.strip() for tag in value.split(",") if tag.strip()]


def init_wandb_run(
    *,
    enabled: bool,
    project: str,
    entity: str | None,
    name: str,
    config: dict[str, Any],
    group: str | None = None,
    job_type: str | None = None,
    tags: list[str] | None = None,
    mode: str = "online",
    run_id: str | None = None,
    sync_tensorboard: bool = False,
    save_code: bool = False,
    step_metric: str = "train/total_timesteps",
) -> Any | None:
    if not enabled:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B logging requested but wandb is not installed. Run: "
            ".venv/bin/pip install -e .[train]"
        ) from exc

    if mode != "disabled" and not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("W&B logging requested but WANDB_API_KEY is not set in the environment.")

    init_kwargs: dict[str, Any] = {
        "project": project,
        "entity": entity or None,
        "name": name,
        "group": group,
        "job_type": job_type,
        "tags": tags or None,
        "config": config,
        "mode": mode,
        "sync_tensorboard": sync_tensorboard,
        "save_code": save_code,
    }
    if run_id:
        init_kwargs["id"] = run_id
        init_kwargs["resume"] = "allow"
    run = wandb.init(**init_kwargs)
    define_wandb_metrics(run, step_metric=step_metric)
    return run


def define_wandb_metrics(run: Any, *, step_metric: str = "train/total_timesteps") -> None:
    run.define_metric(step_metric)
    for pattern in (
        "throughput/*",
        "gpu/*",
        "profile/*",
        "eval/*",
        "eval_profile/*",
        "rollout/*",
        "train/*",
    ):
        run.define_metric(pattern, step_metric=step_metric)


def log_wandb_artifact(
    run: Any | None,
    path: Path,
    *,
    name: str,
    artifact_type: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    if run is None or not path.exists():
        return
    import wandb

    artifact = wandb.Artifact(_artifact_name(name), type=artifact_type, metadata=metadata)
    if path.is_dir():
        artifact.add_dir(str(path))
    else:
        artifact.add_file(str(path))
    run.log_artifact(artifact)


def log_wandb_table(run: Any | None, key: str, records: list[dict[str, Any]]) -> None:
    if run is None or not records:
        return
    import wandb

    columns = list(records[0])
    data = [[record.get(column) for column in columns] for record in records]
    run.log({key: wandb.Table(columns=columns, data=data)})


def finish_wandb_run(run: Any | None, *, status: str | None = None) -> None:
    if run is None:
        return
    if status:
        run.summary["run_status"] = status
    run.finish()


def _artifact_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "artifact"
