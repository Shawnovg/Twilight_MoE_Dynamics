# Attention versus FFN routing: attention_comparison_v1

Configuration: `configs/attention_experiment.yaml`. Seeds: [7, 17, 27]. 40 epochs per model. Every model has current per-run source hashes matching the environment manifest.

## One-step prediction

| Model | Mean test MSE | Sample SD |
|---|---:|---:|
| dense_active | 0.011131 | 0.000212 |
| dense_total | 0.011068 | 0.000165 |
| moe | 0.011466 | 0.000132 |
| random | 0.011705 | 0.000192 |
| oracle | 0.006671 | 0.000045 |
| attn_dense | 0.011013 | 0.000083 |
| attn_active | 0.010961 | 0.000197 |
| attn_total | 0.010813 | 0.000108 |
| attn_moh | 0.010786 | 0.000181 |
| attn_random | 0.010769 | 0.000187 |
| attn_ffn_moe | 0.010677 | 0.000195 |
| attn_ffn_total | 0.010697 | 0.000298 |
| attn_both | 0.010621 | 0.000077 |

Means weight seeds equally. Each seed uses independent data and initialization. SD across three seeds is not a confidence interval.

## Paired comparisons

| First model | Comparator | Mean MSE difference | Mean relative change | First-model wins |
|---|---|---:|---:|---:|
| attn_moh | attn_dense | -0.000228 | -2.07% | 3/3 |
| attn_moh | attn_total | -0.000027 | -0.25% | 2/3 |
| attn_moh | attn_active | -0.000175 | -1.59% | 3/3 |
| attn_moh | attn_random | +0.000017 | +0.16% | 1/3 |
| attn_ffn_moe | attn_dense | -0.000337 | -3.06% | 3/3 |
| attn_ffn_moe | attn_ffn_total | -0.000021 | -0.17% | 2/3 |
| attn_both | attn_ffn_moe | -0.000056 | -0.51% | 1/3 |
| attn_both | attn_moh | -0.000165 | -1.52% | 3/3 |
| attn_moh | dense_total | -0.000282 | -2.55% | 3/3 |
| attn_moh | moe | -0.000680 | -5.94% | 3/3 |
| attn_dense | dense_total | -0.000054 | -0.48% | 2/3 |
| attn_moh | attn_ffn_moe | +0.000109 | +1.03% | 1/3 |

2×2 difference-of-differences (both − MoH − FFN-MoE + dense): +0.000172 MSE. This is descriptive, not a significance test.

## Capacity and training

| Model | Total parameters | Active proxy | Best epochs (seed: epoch) |
|---|---:|---:|---|
| dense_active | 5419 | 5419 | 7: 39, 17: 38, 27: 39 |
| dense_total | 7564 | 7564 | 7: 37, 17: 38, 27: 38 |
| moe | 7599 | 5355 | 7: 37, 17: 40, 27: 39 |
| random | 6828 | 4584 | 7: 40, 17: 40, 27: 40 |
| oracle | 6732 | 2244 | 7: 39, 17: 33, 27: 35 |
| attn_dense | 4628 | 4628 | 7: 39, 17: 40, 27: 40 |
| attn_active | 4116 | 4116 | 7: 39, 17: 38, 27: 40 |
| attn_total | 4826 | 4826 | 7: 39, 17: 38, 27: 40 |
| attn_moh | 4820 | 4820 | 7: 39, 17: 38, 27: 40 |
| attn_random | 4820 | 4820 | 7: 39, 17: 38, 27: 40 |
| attn_ffn_moe | 9012 | 6868 | 7: 40, 17: 39, 27: 40 |
| attn_ffn_total | 8984 | 8984 | 7: 39, 17: 38, 27: 40 |
| attn_both | 9204 | 7060 | 7: 39, 17: 39, 27: 40 |

MoH logically selects 3/4 heads per token, but this reference computes all attention heads. Attention active-parameter proxies include all attention parameters. The FFN path dispatches only selected expert rows. Neither proxy establishes FLOPs or runtime savings.

## Switch recovery and rollouts

| Model | Switch MSE | Recovered fraction | Conditional delay | Mean rollout MSE |
|---|---:|---:|---:|---:|
| dense_active | 0.070248 | 0.764 | 2.652 | 0.057350 |
| dense_total | 0.070517 | 0.762 | 2.652 | 0.057107 |
| moe | 0.069203 | 0.762 | 2.920 | 0.057571 |
| random | 0.069041 | 0.756 | 2.936 | 0.057142 |
| oracle | 0.006916 | 0.816 | 1.480 | 0.054049 |
| attn_dense | 0.069948 | 0.765 | 2.690 | 0.052693 |
| attn_active | 0.070505 | 0.764 | 2.705 | 0.051538 |
| attn_total | 0.071463 | 0.767 | 2.748 | 0.051816 |
| attn_moh | 0.070824 | 0.776 | 2.676 | 0.050773 |
| attn_random | 0.071015 | 0.767 | 2.559 | 0.050855 |
| attn_ffn_moe | 0.071437 | 0.769 | 2.683 | 0.051638 |
| attn_ffn_total | 0.071034 | 0.767 | 2.625 | 0.051934 |
| attn_both | 0.071928 | 0.778 | 2.507 | 0.050514 |

## Interpretation

MoH has lower mean one-step error than standard attention: 0.010786 versus 0.011013. The mean paired relative change is -2.07%, with lower error in 3/3 seeds.

Against the MoH total-parameter dense control, the mean paired relative change is -0.25%. Against the frozen-router control it is +0.16%, with learned MoH winning 1/3 seeds. These controls do not establish a specific benefit from learning the head gate.

35/39 checkpoints were selected within the last three epochs. This is a fixed-training-budget comparison, not evidence that every model has reached its best attainable performance.

A benefit against the old flattened MLP alone would not identify a routing benefit. The standard-attention, total-parameter and frozen-router controls are essential. The attention/FFN 2×2 comparison changes capacity as part of adding FFN experts, so read the FFN total-parameter control too. A comparison of two routing locations is not an isolated test of one psychological concept of feature gathering versus post-processing.

Recovery uses model-specific validation-derived thresholds and conditional delays. Oracle rollout sees future true regimes. Three seeds, one task, one context length and one training preset cannot establish general superiority. No test-driven hyperparameter search was performed.

## Sources and artifacts

- [MoH, Jin et al., ICML 2025](https://proceedings.mlr.press/v267/jin25l.html).
- [MoA, Zhang et al., EMNLP 2022](https://aclanthology.org/2022.emnlp-main.278/).
- Implementation and deviations: `docs/ATTENTION_EXPERIMENT.md`.
- Metrics, histories, diagnostics and hashes: `results\metrics\attention_comparison_v1_9e64af3e67c9`.
- Runnable comparison: notebooks 08–10. Datasets remain on E:.
