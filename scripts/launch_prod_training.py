from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]


REGIMES: dict[str, str] = {
    "mixed": "easy,medium,medium,hard",
    "medium_hard": "medium,medium,hard,hard",
    "hard_focus": "medium,hard,hard,hard",
    "hard": "hard",
}


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _command_text(command: list[str], env: dict[str, str]) -> str:
    prefix = []
    if "CUDA_VISIBLE_DEVICES" in env:
        prefix.append(f"CUDA_VISIBLE_DEVICES={shlex.quote(env['CUDA_VISIBLE_DEVICES'])}")
    return " ".join([*prefix, *(shlex.quote(part) for part in command)])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch grouped production W&B training jobs.")
    parser.add_argument("--regime", choices=sorted(REGIMES), default="medium_hard")
    parser.add_argument("--gpus", default="0", help="Comma-list. Use 0,1 for two parallel jobs.")
    parser.add_argument("--seeds", default="45", help="Comma-list matched cyclically to GPU jobs.")
    parser.add_argument("--workers-per-run", type=int, default=128)
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--group", default=None)
    parser.add_argument("--run-prefix", default=None)
    parser.add_argument("--load", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--checkpoint-freq", type=int, default=500_000)
    parser.add_argument("--status-interval", type=float, default=5.0)
    parser.add_argument("--eval-profile", default="easy,medium,hard")
    parser.add_argument("--extra-tags", default="")
    parser.add_argument("--execute", action="store_true", help="Actually launch jobs. Omitted means print commands only.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    gpus = _split_csv(args.gpus)
    seeds = [int(seed) for seed in _split_csv(args.seeds)]
    if not gpus:
        raise ValueError("--gpus must not be empty")
    if not seeds:
        raise ValueError("--seeds must not be empty")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    group = args.group or f"{args.regime}_{stamp}"
    prefix = args.run_prefix or args.regime
    commands: list[tuple[list[str], dict[str, str]]] = []
    for job_id, gpu in enumerate(gpus):
        seed = seeds[job_id % len(seeds)]
        run_name = f"{prefix}_gpu{gpu}_w{args.workers_per_run}_seed{seed}_{stamp}"
        tags = ["prod", args.regime, f"gpu{gpu}", *(_split_csv(args.extra_tags))]
        command = [
            args.python,
            str(REPO_ROOT / "scripts" / "train_maskable_ppo.py"),
            "--wandb",
            "--wandb-group", group,
            "--wandb-tags", ",".join(tags),
            "--run-name", run_name,
            "--profile", REGIMES[args.regime],
            "--eval-profile", args.eval_profile,
            "--workers", str(args.workers_per_run),
            "--total-timesteps", str(args.total_timesteps),
            "--device", "cuda",
            "--n-steps", str(args.n_steps),
            "--batch-size", str(args.batch_size),
            "--n-epochs", str(args.n_epochs),
            "--learning-rate", str(args.learning_rate),
            "--ent-coef", str(args.ent_coef),
            "--target-kl", str(args.target_kl),
            "--checkpoint-freq", str(args.checkpoint_freq),
            "--eval-freq", "0",
            "--status-interval", str(args.status_interval),
            "--monitor-dir", "runs/monitor",
        ]
        if args.load is not None:
            command.extend(["--load", str(args.load)])
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        commands.append((command, env))

    for command, env in commands:
        print(_command_text(command, env), flush=True)
    if not args.execute:
        return

    processes = [subprocess.Popen(command, cwd=REPO_ROOT, env=env) for command, env in commands]
    failures = 0
    try:
        for process in processes:
            failures += int(process.wait() != 0)
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
        raise
    if failures:
        raise SystemExit(f"{failures} training job(s) failed")


if __name__ == "__main__":
    main()
