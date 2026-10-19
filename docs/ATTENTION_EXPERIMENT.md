# Attention-head routing versus FFN routing

This extension tests the user's hypothesis that routing temporal information
through attention heads could be more useful than routing the flattened history
through MLP experts. The earlier negative result does not establish why FFN MoE
failed. Attention mixes information between positions, while position-wise FFNs
transform each token representation, but both participate in learned feature
construction. The distinction motivates an experiment rather than a conclusion.

## Papers and scope

- **MoH: Multi-Head Attention as Mixture-of-Head Attention**, Peng Jin, Bo Zhu,
  Li Yuan and Shuicheng Yan, ICML 2025.
  [Published paper](https://proceedings.mlr.press/v267/jin25l.html),
  [method equations](https://arxiv.org/html/2410.11842v3),
  [official repository](https://github.com/SkyworkAI/MoH).
  The method combines always-active shared heads with token-selected heads and
  weights their outputs. Its two-stage gate controls head type and individual
  head importance. The authors evaluate vision classification, image generation
  and language models. We adapt its routing mechanism to a tiny causal predictor.
- **Mixture of Attention Heads: Selecting Attention Heads Per Token**, Xiaofeng
  Zhang, Yikang Shen, Zeyu Huang, Jie Zhou, Wenge Rong and Zhang Xiong, EMNLP 2022.
  [Paper](https://aclanthology.org/2022.emnlp-main.278/).
  MoA selects a subset of parameterized attention heads for each token and combines
  their outputs, with experiments on translation and masked language modeling.
  It is an earlier related design, not a separately reproduced model here.

The new model is called **MoH-inspired**. We do not reproduce either paper's
training scale, datasets or performance claims. In particular, selecting heads
in the model equations is different from implementing efficient sparse kernels.

## Predefined experiment

The full preset is `configs/attention_experiment.yaml`. The smoke preset is
`configs/attention_smoke.yaml`. Both include all five old families and eight new
families. The full study uses seeds 7, 17 and 27, 40 epochs, batch size 512,
Adam with learning rate 0.001, gradient clipping at 5 and two CPU threads.
The existing dataset generator is unchanged, so DATA_VERSION remains 1.
All generated datasets stay under `E:/HarunoKen/Twilight_MoE_Dynamics`.

Per seed there are 256/64/64 train/validation/test sequences, 192 transitions per
sequence, 8 observed states per input, 4 coordinates and 3 hidden regimes.
That gives 47,360 training windows and 11,840 windows in each evaluation split.
Splits precede window construction. Only training states fit normalization.
Every non-oracle receives the same observed information. No regime labels enter
an attention family. The original oracle is the sole privileged predictor.

## Common attention backbone

Each input reshapes from `[batch,32]` to `[batch,8,4]`. A linear embedding maps
each state to width 16 and adds a learned position embedding. Two residual
pre-norm Transformer blocks operate on these tokens:

1. LayerNorm, causal self-attention, residual addition.
2. LayerNorm, position-wise GELU FFN, residual addition.

A final LayerNorm and linear readout of the last token predict the next 4-D
state. Q/K/V and output attention projections have no biases. Each head has
dimension 4. FFNs normally use hidden width 32. No dropout or positional
information from beyond the input window is used. Each layer has separate
parameters. The final prediction is one-step, not a sequence of all-token targets.

Each parameter initializes deterministically from the run seed and its full
parameter name. Identical components in the dense, MoH, frozen-router, FFN-MoE
and combined variants therefore start with exactly equal tensors. All families
use the same shuffled window order for a given seed. The original models retain
their original initialization and reproduce their previous results.

## MoH mechanism in this implementation

For each query token, normalized hidden state `z` supplies two linear gates:
`p = softmax(W_route z)` over three routed heads, and
`alpha = softmax(W_type z)` over shared/routed types. A shared-head softmax is
also present. With one shared head, that softmax is identically one.

The shared head receives weight `alpha_shared`. The two selected routed heads
receive `alpha_routed * p[h]`; the third receives zero. We follow the paper's
displayed Equation 5, retaining full-softmax probabilities after masking.
Selected routed weights are **not renormalized**. Thus the retained total head
weight may be less than one. Head outputs receive their separate slices of the
output projection and sum. Masking concatenated head outputs before the common
linear projection is algebraically equivalent.

The attention balance penalty is `sum(mean(p[h]) * fraction_selected[h])` over
routed heads. Fraction selected counts tokens, rather than top-k slots. It is
detached from gradients. The training coefficient is 0.01. Penalties average
over the two blocks. The combined model adds its FFN penalty as well.

All head attention calculations and Q/K/V projections are dense. Gates create
logical head sparsity, not computational sparsity in this reference. We report
3/4 logical heads selected but do not claim reduced runtime, memory or FLOPs.
The attention active-parameter proxy includes every computed head.

## Parallel FFN mechanism

The Transformer FFN-MoE substitutes three `16 → 32 → 16` GELU experts for the
single FFN. A linear token gate selects two experts. Their selected softmax
weights renormalize to one. Sparse row dispatch evaluates each expert on its
selected tokens only. The FFN penalty uses the original project's convention:
three times the dot product of mean probabilities and selected-slot fractions.
These conventions differ from the MoH paper's attention penalty. The study
compares specified routing implementations, not an equal-strength penalty sweep.

The original `moe` baseline is a separate flattened-history MLP mixture. It is
not a Transformer FFN. The new `attn_ffn_moe` gives the same-backbone comparison
needed to distinguish attention-head routing from FFN routing.

The frozen-gate control is also important for interpreting a positive result.
It changes head weights and representation learning even though its gate
parameters do not learn. A gain over standard attention alone therefore does
not establish that learning head selection caused the improvement.

## Model families and what they control

| Model | Attention | FFN | Purpose |
|---|---|---|---|
| attn_dense | 4 standard heads | Dense | Common-backbone baseline |
| attn_moh | 1 shared + top-2 of 3 routed | Dense | Attention-routing treatment |
| attn_ffn_moe | 4 standard heads | 3 experts, top-2 | FFN-routing treatment |
| attn_both | MoH | FFN-MoE | Both routing locations |
| attn_random | Same gate structure as MoH, frozen weights | Dense | Learned-gate ablation |
| attn_active | 3 standard heads of dimension 4 | Dense | Same count of contributing heads |
| attn_total | 4 standard heads | Width chosen to match MoH total params | Added-parameter control |
| attn_ffn_total | 4 standard heads | Width chosen to match FFN-MoE total params | FFN capacity control |

The three-head model keeps embedding width 16 and head width 4, with an internal
attention width of 12. It is a logical head-count control, not a runtime match.
The total-parameter controls change FFN hidden width to the nearest match.
The random gate has frozen parameters but remains input-dependent. Its input
representations still change during training, so routes need not stay fixed.

The original five families remain dense_active, dense_total, moe, random and
oracle. Cross-backbone comparisons with these are descriptive: they differ in
normalization layers, activation functions, parameterization and initialization.

## Outcomes and interpretation

Primary outcome: original-unit test one-step MSE, averaged over windows and four
coordinates. The checkpoint is the epoch with the lowest normalized validation
prediction MSE. Auxiliary penalties do not enter checkpoint selection.

Primary contrast: MoH minus standard attention, paired by seed. Additional
contrasts use the frozen gate, head-count and total-parameter controls. The 2×2
common-backbone comparison also reports FFN-MoE versus dense, and both versus
each single routing location. No test-driven parameter search is performed.

Secondary outcomes use the unchanged evaluation code: switch-aligned error,
model-specific recovery fraction and conditional delay, and recursive horizon
1–32 MSE. Delays must be read with the recovered fraction. Late switch lags have
lower support. Oracle rollouts receive true future regimes and remain privileged.

Attention diagnostics show final-query route utilization, validation-mapped
regime agreement, raw regime/head counts and mean attention over history.
Shared heads are excluded from regime matching. The three routed head IDs are
exchangeable. Mean attention plots include heads that were gated off for some
tokens, so these plots describe ungated attention distributions, not causal
importance. First-block routing sees only a token and its position; routing in
the second block can use contextual information from the first block.

## Observed full-run outcome

The completed three-seed experiment gives mean one-step MSE 0.011013 for
standard attention, 0.010786 for MoH, 0.010677 for Transformer FFN-MoE and
0.010621 for both routing locations. MoH improves over standard attention in
all three seeds, with mean paired relative change -2.07%. However, its gain
over the total-parameter dense control is only 0.25%, and the frozen-router
control has slightly lower mean error (0.010769). Learned head routing is
therefore not established as the cause of the improvement.

Transformer FFN-MoE has lower mean one-step error than MoH, although it has
more parameters. Its gain over its own total-parameter dense control is only
0.17%. The combined model has the lowest non-oracle mean error, but beats
FFN-MoE in only one of three seeds. These results do not establish attention
routing as superior to FFN routing on this task.

MoH has lower mean 32-step rollout error than standard attention (0.050773
versus 0.052693), while its error at a regime switch is slightly higher
(0.070824 versus 0.069948). The benefits depend on the metric. Most selected
checkpoints are near the 40-epoch budget, so the study compares a fixed budget,
not fully optimized model families. See the generated result log for every
control, seed spread and checkpoint epoch.

## Reproduction and function reference

Implementation entry points:

| Function/class | Role |
|---|---|
| `HeadAttention.gates` | Shared/routed weights, top-k IDs and attention balancing loss |
| `HeadAttention.forward` | Causal attention and gated head output projection |
| `TokenFFN.forward` | Dense FFN or sparse dispatch of token-wise expert rows |
| `TransformerBlock.forward` | Pre-norm attention/FFN residual block |
| `AttentionPredictor.reset_seeded` | Paired, deterministic component initialization |
| `AttentionPredictor.forward_features` | Token embedding, blocks and optional diagnostics |
| `AttentionPredictor.forward` | Last-token 4-D prediction with the common model API |
| `AttentionPredictor.capacity` | Parameter counts and explicit dense-attention compute flag |
| `collect_attention` | Batched head weights, routes and mean final-query attention |
| `evaluate_attention` | Validation-mapped head/FFN regime agreement and utilization |
| `configured_models` | Choose the experiment's model-family list |
| `verify_provenance` | Check completeness and hashes for every reported family/seed |

Constructors build the corresponding modules from the configuration. The existing
`build_model`, `fit` and `run_models` now dispatch these variants while preserving
the original families. New scripts generate notebooks and independently verify
saved checkpoints. `summarize_results.py` writes the attention-specific readout
when the configuration includes an attention section.

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts/run_experiment.py --config configs/attention_experiment.yaml
.\.venv\Scripts\python.exe scripts/execute_notebooks.py --config configs/attention_smoke.yaml --output-dir results/notebook_smoke_attention
.\.venv\Scripts\python.exe scripts/execute_notebooks.py --config configs/attention_experiment.yaml --pattern 0[89]*.ipynb
.\.venv\Scripts\python.exe scripts/execute_notebooks.py --config configs/attention_experiment.yaml --pattern 10*.ipynb
.\.venv\Scripts\python.exe scripts/summarize_results.py --config configs/attention_experiment.yaml
```

Notebook 08 reuses a complete current-source cache unless `FORCE_RETRAIN=True`.
Notebooks 09–10 refuse incomplete or stale comparisons. Each result JSON and
checkpoint saves source hashes in addition to the experiment environment
manifest. This avoids treating one overwritten manifest as proof that all
families use the same code. Generator checksums remain unchanged.

Results are written to `results/attention_comparison_log.md` and the experiment's
`comparison.md`. The consolidated technical report now covers both experiments.
The updated deck is `presentations/Twilight_MoE_Dynamics_Research_Updated.pptx`.
The original deck and `docs/archive/TECHNICAL_REPORT_2026-10-08.md` preserve
the earlier presentation and report.
