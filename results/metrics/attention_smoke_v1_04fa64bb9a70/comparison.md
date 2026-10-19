# Attention versus FFN routing: attention_smoke_v1

Configuration: `configs/attention_smoke.yaml`. Seeds: [7]. 6 epochs per model. Every model has current per-run source hashes matching the environment manifest.

## One-step prediction

| Model | Mean test MSE | Sample SD |
|---|---:|---:|
| dense_active | 0.019597 | nan |
| dense_total | 0.018831 | nan |
| moe | 0.024244 | nan |
| random | 0.025156 | nan |
| oracle | 0.013020 | nan |
| attn_dense | 0.031860 | nan |
| attn_active | 0.030154 | nan |
| attn_total | 0.030956 | nan |
| attn_moh | 0.032350 | nan |
| attn_random | 0.032318 | nan |
| attn_ffn_moe | 0.031826 | nan |
| attn_ffn_total | 0.029345 | nan |
| attn_both | 0.031195 | nan |

Means weight seeds equally. Each seed uses independent data and initialization. SD across three seeds is not a confidence interval.

## Paired comparisons

| First model | Comparator | Mean MSE difference | Mean relative change | First-model wins |
|---|---|---:|---:|---:|
| attn_moh | attn_dense | +0.000490 | +1.54% | 0/1 |
| attn_moh | attn_total | +0.001394 | +4.50% | 0/1 |
| attn_moh | attn_active | +0.002196 | +7.28% | 0/1 |
| attn_moh | attn_random | +0.000032 | +0.10% | 0/1 |
| attn_ffn_moe | attn_dense | -0.000034 | -0.11% | 1/1 |
| attn_ffn_moe | attn_ffn_total | +0.002481 | +8.45% | 0/1 |
| attn_both | attn_ffn_moe | -0.000631 | -1.98% | 1/1 |
| attn_both | attn_moh | -0.001155 | -3.57% | 1/1 |
| attn_moh | dense_total | +0.013520 | +71.80% | 0/1 |
| attn_moh | moe | +0.008106 | +33.43% | 0/1 |
| attn_dense | dense_total | +0.013029 | +69.19% | 0/1 |

2×2 difference-of-differences (both − MoH − FFN-MoE + dense): -0.001121 MSE. This is descriptive, not a significance test.

## Capacity and training

| Model | Total parameters | Active proxy | Best epochs (seed: epoch) |
|---|---:|---:|---|
| dense_active | 3612 | 3612 | 7: 6 |
| dense_total | 5119 | 5119 | 7: 6 |
| moe | 5055 | 3563 | 7: 6 |
| random | 4572 | 3080 | 7: 6 |
| oracle | 4476 | 1492 | 7: 6 |
| attn_dense | 4628 | 4628 | 7: 6 |
| attn_active | 4116 | 4116 | 7: 6 |
| attn_total | 4826 | 4826 | 7: 6 |
| attn_moh | 4820 | 4820 | 7: 6 |
| attn_random | 4820 | 4820 | 7: 6 |
| attn_ffn_moe | 9012 | 6868 | 7: 6 |
| attn_ffn_total | 8984 | 8984 | 7: 6 |
| attn_both | 9204 | 7060 | 7: 6 |

MoH logically selects 3/4 heads per token, but this reference computes all attention heads. Attention active-parameter proxies include all attention parameters. The FFN path dispatches only selected expert rows. Neither proxy establishes FLOPs or runtime savings.

## Switch recovery and rollouts

| Model | Switch MSE | Recovered fraction | Conditional delay | Mean rollout MSE |
|---|---:|---:|---:|---:|
| dense_active | 0.056861 | 0.703 | 3.846 | 0.056728 |
| dense_total | 0.058829 | 0.703 | 3.731 | 0.055930 |
| moe | 0.055963 | 0.730 | 4.333 | 0.056489 |
| random | 0.056253 | 0.730 | 4.111 | 0.057687 |
| oracle | 0.030651 | 0.811 | 1.967 | 0.044835 |
| attn_dense | 0.042139 | 0.730 | 1.296 | 0.067887 |
| attn_active | 0.041253 | 0.676 | 2.680 | 0.060680 |
| attn_total | 0.039289 | 0.649 | 1.583 | 0.061366 |
| attn_moh | 0.041294 | 0.703 | 1.385 | 0.069323 |
| attn_random | 0.040933 | 0.703 | 1.385 | 0.069233 |
| attn_ffn_moe | 0.044251 | 0.676 | 1.760 | 0.063791 |
| attn_ffn_total | 0.039994 | 0.703 | 2.308 | 0.068327 |
| attn_both | 0.040561 | 0.649 | 1.958 | 0.065420 |

## Interpretation

MoH has higher mean one-step error than standard attention: 0.032350 versus 0.031860. The mean paired relative change is +1.54%, with lower error in 0/1 seeds.

A benefit against the old flattened MLP alone would not identify a routing benefit. The standard-attention, total-parameter and frozen-router controls are essential. The attention/FFN 2×2 comparison changes capacity as part of adding FFN experts, so read the FFN total-parameter control too. A comparison of two routing locations is not an isolated test of one psychological concept of feature gathering versus post-processing.

Recovery uses model-specific validation-derived thresholds and conditional delays. Oracle rollout sees future true regimes. Three seeds, one task, one context length and one training preset cannot establish general superiority. No test-driven hyperparameter search was performed.

## Sources and artifacts

- [MoH, Jin et al., ICML 2025](https://proceedings.mlr.press/v267/jin25l.html).
- [MoA, Zhang et al., EMNLP 2022](https://aclanthology.org/2022.emnlp-main.278/).
- Implementation and deviations: `docs/ATTENTION_EXPERIMENT.md`.
- Metrics, histories, diagnostics and hashes: `results\metrics\attention_smoke_v1_04fa64bb9a70`.
- Runnable comparison: notebooks 08–10. Datasets remain on E:.
