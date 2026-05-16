# Production Training Pipeline

This is the scriptable path for hosted training runs. The notebook is still useful for interactive experiments, but production runs should be resumable, manifest-backed, and evaluated out of process.

## Current Learnability Baseline

The first medium run reached 2M timesteps with `workers=96`, `device=cuda`, and army `barbarian=40,wall_breaker=10`.

200-episode medium holdout:

- `score_mean=2.996`
- `damage_mean=0.791`
- `stars_mean=2.205`
- `p_stars_ge_2=0.885`
- `p_damage_ge_90=0.440`
- `3-star=67/200`

50-episode cross-profile eval:

| Eval profile | Score mean | Damage mean | Stars mean | Readout |
| --- | ---: | ---: | ---: | --- |
| easy | `2.108` | `0.668` | `1.440` | below random baseline |
| medium | `3.107` | `0.807` | `2.300` | strong in-distribution |
| hard | `0.488` | `0.288` | `0.200` | only marginally above random |

This is a real learnability proof, but not a profile-agnostic attacker. The next production run should train on a mixed curriculum from the start rather than relying on medium-only transfer.

## Training

Use `scripts/train_maskable_ppo.py` for production runs. It now writes:

- `checkpoints/maskable_ppo/<run_name>/run_config.json`
- `checkpoints/maskable_ppo/<run_name>/run_status.json`
- `checkpoints/maskable_ppo/<run_name>/checkpoint_*_steps.zip`
- `checkpoints/maskable_ppo/<run_name>/final_model.zip`
- `checkpoints/maskable_ppo/<run_name>/interrupted_model.zip` on Ctrl-C

Serial in-loop eval is disabled by default. Use offline evaluation unless you explicitly need a small sentinel eval during training.

Example curriculum run:

```bash
.venv/bin/python scripts/train_maskable_ppo.py   --run-name curriculum_mmh_w128_seed43_20260515   --profile medium,medium,hard   --workers 128   --total-timesteps 5000000   --device cuda   --n-steps 512   --batch-size 4096   --checkpoint-freq 500000   --eval-freq 0   --monitor-dir runs/monitor
```

Resume a run:

```bash
.venv/bin/python scripts/train_maskable_ppo.py   --run-name curriculum_mmh_w128_seed43_20260515   --resume   --profile medium,medium,hard   --workers 128   --total-timesteps 5000000   --device cuda   --n-steps 512   --batch-size 4096   --checkpoint-freq 500000   --eval-freq 0   --monitor-dir runs/monitor
```

Validate a config without starting workers:

```bash
.venv/bin/python scripts/train_maskable_ppo.py   --run-name dry_run   --profile medium,hard   --workers 96   --total-timesteps 1000   --device cuda   --dry-run
```

## Offline Evaluation

Use `scripts/evaluate_maskable_ppo.py`. Prefer `--device cpu` for parallel eval so multiple workers do not fight over one GPU context.

Medium holdout:

```bash
.venv/bin/python scripts/evaluate_maskable_ppo.py   --checkpoint checkpoints/maskable_ppo/medium_w96_cuda_seed42_20260515_205523/final_model.zip   --profile medium   --episodes 200   --seed-start 100000   --workers 8   --device cpu   --output-dir runs/eval/medium_w96_cuda_seed42_20260515_205523   --name medium_holdout_parallel
```

Cross-profile eval:

```bash
for profile in easy medium hard; do
  .venv/bin/python scripts/evaluate_maskable_ppo.py     --checkpoint checkpoints/maskable_ppo/medium_w96_cuda_seed42_20260515_205523/final_model.zip     --profile "$profile"     --episodes 100     --seed-start 200000     --workers 8     --device cpu     --output-dir runs/eval/medium_w96_cuda_seed42_20260515_205523     --name "cross_profile_${profile}_parallel"
done
```

## Next Engineering Bottleneck

Training throughput is still rollout-bound, not GPU-bound. The next high-leverage implementation work is:

1. Add flat training observations while keeping dict observations for viewer/debugging.
2. Vectorize and cache deploy masks by terrain version.
3. Add simulator training fast-mode that skips viewer-only visual events.
4. If needed, replace the SB3 mask-fetch path with a rollout collector that returns next action masks with observations.
