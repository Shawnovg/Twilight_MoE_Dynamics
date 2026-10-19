# Experiment plan

## Question and scope

Test whether conditional computation helps next-state prediction in a controlled
switching linear system: x[t+1] = A[r[t]] x[t] + Gaussian noise, with four state
coordinates and three hidden Markov regimes. No performance advantage is assumed.
SAE and DAS are future interventions, outside version 1.

## Implementation sequence

1. Generate reproducible train/validation/test sequences with provenance hashes.
2. Construct causal history windows and training-only normalization.
3. Implement dense active/total parameter controls, learned sparse MoE, fixed
   random router, and regime-informed oracle.
4. Train with shared settings and validation checkpoint selection.
5. Evaluate prediction, rollout, switch recovery, capacity, and routing.
6. Run unit tests and all eight notebooks, inspect figures, record results.

## Assumptions and interpretation

- Each transition matrix is a contracted orthogonal matrix with operator norm
  0.94. This ensures contraction under arbitrary switching, stronger than
  checking each matrix's spectral radius alone. Noise prevents collapse to zero.
- Regimes persist with probability 0.95 by default. Regime r[t] produces the
  transition from x[t] to x[t+1]. The next regime is sampled independently of
  the current state conditional on the previous regime. A new switch cannot
  always be inferred before its first consequence is observed.
- A context of eight states offers transition evidence. Labels never enter the
  learned or random router. Three-layer expert MLPs have two hidden layers.
- Learned routing uses top-2 by default so selected softmax weights have task
  gradients. Top-1 uses a documented straight-through gate estimator. A small
  load-balancing auxiliary term discourages collapse. Only selected experts
  are executed. The random control uses a seeded frozen random linear router,
  deterministic for each input and unchanged throughout training/evaluation.
- The two dense widths minimize absolute mismatch to total and active parameter
  counts. Active count includes the whole router plus k experts; this is a
  parameter proxy, not measured FLOPs, latency, or training cost. Report mismatch.
- Shared epochs, optimizer, batches, and validation selection are the initial
  fairness controls; hyperparameter tuning budgets are not yet explored.
- Oracle routing selects the true-regime expert. It has different information
  and active capacity from top-2 MoE. A known-matrix oracle isolates the generator
  noise reference from optimization limitations of the trained oracle.
- Multi-step rollouts feed back predictions without future observations. The
  oracle receives future true regimes and must be interpreted accordingly.
- Switch recovery is defined using validation data: three consecutive errors
  below 1.5 times that model's late-regime validation mean MSE. Episodes end at
  the next switch or sequence end. Report recovered fraction and delays among
  recovered episodes; unrecovered episodes are censored, never treated as zero.
  Also plot event-aligned error with counts to expose selection/survival effects.
- Expert/regime agreement uses a permutation fitted on validation only. It is
  diagnostic and does not establish causal mechanisms or interpretability.
- The full preset varies both dataset and initialization across three seeds.
  Report paired differences within seeds and their spread. Three seeds are a
  starting point, not strong statistical evidence or a tuned benchmark.

## Later work

Expand seed counts, context lengths, switch rates, noise levels, expert counts,
and matched tuning budgets before asserting an advantage. Add nonlinear
oscillators and one small real dataset only after validating the synthetic
pipeline. Then explore recurrent routing, adaptive stopping, and SAE/DAS
interventions with explicit causal controls and reconstruction checks.
