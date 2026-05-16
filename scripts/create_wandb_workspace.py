from __future__ import annotations

import argparse
import os
from typing import Any


def _resolve_entity(explicit: str | None) -> str:
    if explicit:
        return explicit
    if os.environ.get("WANDB_ENTITY"):
        return os.environ["WANDB_ENTITY"]
    try:
        import wandb

        viewer = wandb.Api().viewer()
    except Exception as exc:
        raise RuntimeError("Pass --entity or set WANDB_ENTITY so the workspace owner is explicit.") from exc
    entity = getattr(viewer, "entity", None) or getattr(viewer, "username", None)
    if not entity:
        raise RuntimeError("Could not infer W&B entity; pass --entity or set WANDB_ENTITY.")
    return str(entity)


def _line(wr: Any, *metrics: str) -> Any:
    return wr.LinePlot(x="train/total_timesteps", y=list(metrics))


def _scalar(wr: Any, metric: str) -> Any:
    return wr.ScalarChart(metric=metric, groupby_aggfunc="max")


def build_workspace(*, entity: str, project: str, name: str) -> Any:
    try:
        import wandb_workspaces.reports.v2 as wr
        import wandb_workspaces.workspaces as ws
    except ImportError as exc:
        raise RuntimeError(
            "wandb-workspaces is not installed. Run: .venv/bin/pip install -e .[train]"
        ) from exc

    sections = [
        ws.Section(
            name="Executive Scorecard",
            panels=[
                _scalar(wr, "eval_profile/easy/score_mean"),
                _scalar(wr, "eval_profile/medium/score_mean"),
                _scalar(wr, "eval_profile/hard/score_mean"),
                _scalar(wr, "eval_profile/hard/p_stars_ge_2"),
            ],
            is_open=True,
        ),
        ws.Section(
            name="Evaluation Quality",
            panels=[
                _line(
                    wr,
                    "eval_profile/easy/score_mean",
                    "eval_profile/medium/score_mean",
                    "eval_profile/hard/score_mean",
                ),
                _line(
                    wr,
                    "eval_profile/easy/damage_mean",
                    "eval_profile/medium/damage_mean",
                    "eval_profile/hard/damage_mean",
                ),
                _line(
                    wr,
                    "eval_profile/easy/p_stars_ge_2",
                    "eval_profile/medium/p_stars_ge_2",
                    "eval_profile/hard/p_stars_ge_2",
                ),
                _line(
                    wr,
                    "eval_profile/easy/p_damage_ge_90",
                    "eval_profile/medium/p_damage_ge_90",
                    "eval_profile/hard/p_damage_ge_90",
                ),
            ],
            is_open=True,
        ),
        ws.Section(
            name="Rollout Windows",
            panels=[
                _line(wr, "profile/easy/score_mean", "profile/medium/score_mean", "profile/hard/score_mean"),
                _line(wr, "profile/easy/damage_mean", "profile/medium/damage_mean", "profile/hard/damage_mean"),
                _line(wr, "profile/easy/stars_mean", "profile/medium/stars_mean", "profile/hard/stars_mean"),
                _line(wr, "profile/easy/truncated_rate", "profile/medium/truncated_rate", "profile/hard/truncated_rate"),
            ],
            is_open=True,
        ),
        ws.Section(
            name="Throughput And Hardware",
            panels=[
                _line(wr, "throughput/env_steps_per_sec", "throughput/episodes_per_sec"),
                _line(wr, "gpu/gpu_util_pct_mean", "gpu/gpu_util_pct_max"),
                _line(wr, "gpu/gpu_mem_used_gb_mean", "gpu/gpu_mem_total_gb_mean"),
            ],
            is_open=True,
        ),
        ws.Section(
            name="PPO Health",
            panels=[
                _line(wr, "train/approx_kl", "train/clip_fraction"),
                _line(wr, "train/entropy_loss"),
                _line(wr, "train/value_loss", "train/explained_variance"),
                _line(wr, "train/policy_gradient_loss", "train/loss"),
            ],
            is_open=True,
        ),
        ws.Section(
            name="Hard Profile Diagnosis",
            panels=[
                _line(wr, "profile/hard/score_mean", "eval_profile/hard/score_mean"),
                _line(wr, "profile/hard/p_damage_ge_90", "eval_profile/hard/p_damage_ge_90"),
                _line(wr, "profile/hard/p_stars_ge_2", "eval_profile/hard/p_stars_ge_2"),
                _line(wr, "profile/hard/ep_len_mean", "profile/hard/ticks_mean", "profile/hard/truncated_rate"),
            ],
            is_open=True,
        ),
    ]
    return ws.Workspace(entity=entity, project=project, name=name, sections=sections)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create the ClashAI production W&B workspace.")
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "clashai-rl"))
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--name", default="ClashAI RL Production")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    entity = _resolve_entity(args.entity)
    workspace = build_workspace(entity=entity, project=args.project, name=args.name)
    saved = workspace.save()
    print(getattr(workspace, "url", None) or saved or "workspace saved", flush=True)


if __name__ == "__main__":
    main()
