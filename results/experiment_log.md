# Experiment log: experiment

Configuration: `configs/experiment.yaml`. Seeds: [7, 17, 27]. 40 epochs per model; CPU; validation-selected checkpoints.

| Model | Mean test MSE | Sample SD across seeds |
|---|---:|---:|
| dense_active | 0.011131 | 0.000212 |
| dense_total | 0.011068 | 0.000165 |
| moe | 0.011466 | 0.000132 |
| oracle | 0.006671 | 0.000045 |
| random | 0.011705 | 0.000192 |

MSE is averaged over four coordinates in original state units squared. Seed spread is not a confidence interval. Seeds vary both dataset and initialization.

## Paired comparisons

- Against dense_active, learned MoE mean paired MSE difference is +0.000335 (sample SD 0.000146); mean within-seed relative change +3.03%. MoE has lower error in 0/3 seeds.
- Against dense_total, learned MoE mean paired MSE difference is +0.000399 (sample SD 0.000106); mean within-seed relative change +3.61%. MoE has lower error in 0/3 seeds.

## Capacity and checkpoint selection

| Model | Total parameters | Active parameter proxy | Selected epochs by seed |
|---|---:|---:|---|
| dense_active | 5419 | 5419 | 7: 39, 17: 38, 27: 39 |
| dense_total | 7564 | 7564 | 7: 37, 17: 38, 27: 38 |
| moe | 7599 | 5355 | 7: 37, 17: 40, 27: 39 |
| oracle | 6732 | 2244 | 7: 39, 17: 33, 27: 35 |
| random | 6828 | 4584 | 7: 40, 17: 40, 27: 40 |

## Interpretation and limits

Persistence MSE averaged over seeds: 0.039325. Known dynamics with true regime: 0.006396; configured noise variance: 0.006400.

The trained oracle receives privileged regime labels. Its error is a reference, not a guaranteed optimization bound. Parameter proxies do not establish runtime or FLOP savings. Random routing uses a frozen linear gate, with its smaller router included in capacity reporting.

This is one synthetic task and one fixed hyperparameter preset. Three seeds do not establish a general advantage or disadvantage for MoE. Report the result as an initial controlled benchmark; test broader seeds, context lengths, switching rates, and equal tuning budgets before a larger claim.

Switch recovery delays are conditional on recovery; unrecovered episodes are censored. Thresholds are model-specific and validation-derived. Read delays together with recovery fractions. Multi-step oracle rollouts receive future true regimes.

## Reproduction and artifacts

- Metrics and provenance: `results\metrics\experiment_57adae732ea0`.
- Figures: `results/figures/experiment_57adae732ea0`.
- Dataset source and checksums: the configured E: data root, `metadata/`.
- Exact installed package versions: `requirements-lock.txt`.
- Saved notebook outputs were produced by `scripts/execute_notebooks.py`.
- Smoke results are stored separately from the full preset.
- Unit test and visual verification outcomes are recorded in `VERIFICATION.md`.
