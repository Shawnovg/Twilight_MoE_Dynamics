# Twilight MoE Dynamics: technical report

> Historical report for the original MLP experiment, reviewed on 8 October 2026.
> The later attention extension changes package sources, so the source-hash
> statement below records the original review rather than the current checkout.
> For current-source comparisons see `ATTENTION_EXPERIMENT.md` and
> `../results/attention_comparison_log.md`. The original numerical findings are preserved.

**Implementation:** version 0.1.0  
**Review date:** 8 October 2026  
**Code:** `D:/HarunoKen/Twilight_MoE_Dynamics`  
**Data:** `E:/HarunoKen/Twilight_MoE_Dynamics`  
**Main experiment:** `experiment_57adae732ea0`, seeds 7, 17, and 27

This report explains the actual program, including its mathematical assumptions, implementation, saved results, and limitations. Its primary sources are the project code, configuration files, notebooks, and recorded experiment artifacts. The source hashes in the full experiment's environment manifest match the current Python implementation. The numerical summaries below were recomputed from the saved result files during this review; the models were not retrained for this report.

**Reading guide:** Sections 1–3 introduce the question and notation. Sections 4–5 explain the generated data. Sections 6–9 explain model construction and learning. Sections 10–14 define evaluation and configuration. Section 15 interprets the observed results. Sections 16–20 document the notebooks, code, artifacts, tests, and commands. Sections 21–22 discuss research limits and terminology. If you want the conceptual explanation first, read Sections 1, 4.5, 6, and 15 before the implementation details.

## 1. What the project asks

The project asks whether a small Mixture of Experts (MoE) predicts a switching dynamical system better than dense neural networks with comparable capacity.

Imagine observing an object whose movement follows one of three rules. You see its recent positions, but nobody tells you when its movement rule changes. Your task is to predict its next position. A dense model uses the same network for every observation. An MoE has several expert networks and a router that chooses which experts to use for each observation.

The experiment tests whether that division of work helps. It also asks whether the router learns something related to the hidden movement rule, how errors behave immediately after a rule change, and whether one-step predictions remain useful when repeatedly fed back into the model.

The completed full preset did **not** find a one-step advantage for the learned MoE. Its average test MSE was 0.011466, compared with 0.011131 for the dense active-parameter control and 0.011068 for the dense total-parameter control. It had higher error than both dense controls in every tested seed. This is a result for this task and configuration, not a general conclusion about MoE architectures.

The program is a controlled research foundation. It is not a large language model, a time-series foundation model, an SAE experiment, or a demonstrated new method. SAE means sparse autoencoder; DAS means distributed alignment search. Neither is implemented here.

## 2. The complete experiment at a glance

```text
YAML configuration and experiment seed
                  |
                  v
Independent synthetic train / validation / test sequences
                  |
                  v
Training-state mean and standard deviation
                  |
                  v
Causal history windows: 8 observed states -> 32 input numbers
                  |
        +---------+---------+---------+---------+
        |         |         |         |         |
    dense      dense     learned    frozen    regime-informed
    active     total       MoE      random       oracle
        |         |         |         |         |
        +---------+---------+---------+---------+
                  |
                  v
Fixed-epoch training; select best validation checkpoint
                  |
                  v
Held-out one-step errors, routing diagnostics,
switch recovery, and free-running rollouts
                  |
                  v
Metrics / model checkpoints / notebook figures / experiment log
```

Here, **training** changes model parameters, **validation** chooses a checkpoint and defines diagnostic thresholds, and **testing** measures the selected model on independent sequences. Whole sequences are assigned to splits before windows are constructed. This order prevents nearly identical overlapping windows from entering different splits.

There is one MoE module containing three experts. Each expert is a three-layer multilayer perceptron (MLP): two hidden layers and an output layer. There are not three sequential MoE blocks. There is no recurrent hidden state, attention mechanism, repeated internal reasoning loop, or adaptive stopping mechanism.

## 3. Mathematical notation and units

| Symbol | Meaning | Default full-preset value or shape |
|---|---|---|
| \(x_t\) | Observed system state at time index \(t\) | Four real-valued coordinates |
| \(r_t\) | Hidden regime that generates the transition from \(t\) to \(t+1\) | 0, 1, or 2 |
| \(A_r\) | Transition matrix for regime \(r\) | \(4\times4\) |
| \(\epsilon_t\) | Independent process noise | \(\mathcal N(0,0.08^2 I_4)\) |
| \(P\) | Markov regime-transition matrix | \(3\times3\) |
| \(L\) | Number of retained transitions per sequence | 192 |
| \(C\) | Context length in observed states | 8 |
| \(u_t\) | Flattened, normalized context | 32 coordinates |
| \(E\) | Number of experts | 3 |
| \(k\) | Number of selected experts per input | 2, except oracle uses 1 |
| \(B\) | Training batch size | 512 |
| \(H\) | Maximum rollout horizon | 32 |

Coordinates are dimensionless synthetic values. A time step is an index, not a specified number of seconds. MSE is measured in squared original state units unless explicitly labeled **normalized**. Code indexes regimes from zero; describing them as regimes 1, 2, and 3 would be only a relabeling.

## 4. The synthetic dynamical system

### 4.1 State evolution

The generator implements

\[
x_{t+1}=A_{r_t}x_t+\epsilon_t,
\qquad \epsilon_t\sim\mathcal N(0,\sigma^2 I_4),\quad \sigma=0.08.
\]

The state is directly observed. The regime is hidden from ordinary models. Noise is added to the system's evolution; there is no separate observation-noise process.

The order matters. At time \(t\), the generator stores the current \(x_t\) and \(r_t\), generates \(x_{t+1}\) using \(r_t\), and then samples \(r_{t+1}\). Thus the label paired with a training window ending at \(x_t\) is the regime that produces its target \(x_{t+1}\).

### 4.2 Transition matrices

Define a two-dimensional rotation

\[
R(\theta)=\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}.
\]

Each regime matrix is

\[
A_r=0.94\begin{bmatrix}R(a_r)&0\\0&R(b_r)\end{bmatrix}.
\]

| Regime | Rotation of coordinates 0–1, radians | Rotation of coordinates 2–3, radians |
|---|---:|---:|
| 0 | 0.20 | 0.45 |
| 1 | 1.10 | −0.65 |
| 2 | −0.80 | 1.35 |

The two coordinate pairs rotate independently, while the same regime controls both pairs. The contraction factor reduces the noiseless vector length by 6% per step.

For every regime, \(\|A_r\|_2=0.94\). Therefore any noiseless switched trajectory satisfies

\[
\|A_{r_{t-1}}\cdots A_{r_0}x_0\|_2\leq 0.94^t\|x_0\|_2.
\]

This establishes contraction even under arbitrary switching. Merely checking that each individual matrix has eigenvalues inside the unit circle would not provide that guarantee for arbitrary general matrices. The guarantee here comes from the common operator-norm bound.

This construction is deliberately simple: the matrices have a shared block structure, no cross-pair coupling, no additive regime-specific offset, and the same contraction factor. The matrices themselves are fixed across experiment seeds. Seeds change trajectories and neural-network initialization, not the family of dynamical laws.

### 4.3 Regime switching

The Markov transition matrix is

\[
P=\begin{bmatrix}
0.95&0.025&0.025\\
0.025&0.95&0.025\\
0.025&0.025&0.95
\end{bmatrix},
\qquad P_{ij}=\Pr(r_{t+1}=j\mid r_t=i).
\]

The next regime depends on the current regime, not on the current state's value. At every transition, the regime stays the same with probability 0.95. If it changes, each alternative is equally likely.

The initial regime is sampled uniformly. The symmetric transition matrix also has a uniform stationary regime distribution. Consequently, long-run coverage is approximately one-third per regime, although a finite dataset will not have exactly equal counts.

The mean complete dwell length is \(1/(1-0.95)=20\) transitions. This is a distributional mean, not a fixed segment length. Short and long segments both occur, and segments at recorded sequence boundaries are truncated.

### 4.4 Initialization, noise, and burn-in

The initial state is sampled independently in each coordinate with standard deviation

\[
s_x=\frac{\sigma}{\sqrt{1-0.94^2}}\approx0.234484.
\]

The corresponding variance is about 0.054983. This follows from the stationary covariance equation \(v=0.94^2v+\sigma^2\).

Because all regime matrices are scaled orthogonal transformations and the noise is isotropic, this Gaussian covariance is preserved under each regime. In the ideal mathematical construction, the initialization already supplies the stationary isotropic state distribution. The configured burn-in of 128 steps is retained as a conservative simulation convention; this particular construction does not need burn-in to correct an initial covariance mismatch.

Noise prevents the contracting dynamics from simply decaying to zero. It also imposes a genuine prediction limit. Even if a predictor knows the exact matrix and the current regime, it cannot predict the next independent noise realization.

### 4.5 Why history is essential

At stationarity, a single state has the same isotropic marginal distribution under each regime. A large or small coordinate value does not directly identify the regime. The distinguishing evidence is in **how the state changes**.

The eight-state context contains seven observed transitions, governed by labels \(r_{t-7},\ldots,r_{t-1}\). There are six boundaries between those labels, so the probability that these observed transitions involve more than one regime is \(1-0.95^6\approx26.49\%\). Including the target transition's label \(r_t\) adds a seventh switch opportunity. A useful model therefore needs to handle windows containing mixed dynamics.

There is an additional causal limitation. When \(r_t\) changes, its first effect is on \(x_{t+1}\), which the model is trying to predict. The observed history through \(x_t\) contains evidence about earlier regimes, but cannot reveal the realization of a newly sampled regime with certainty. This explains why a non-oracle model can have a sharp error spike at switch lag zero even if it has learned the dynamics well.

These observations are deductions from the implemented generator, not extra experiments or claims about real-world data.

Source: [dynamics.py](D:/HarunoKen/Twilight_MoE_Dynamics/src/twilight_moe/dynamics.py).

## 5. Data organization and preprocessing

### 5.1 Dataset dimensions

Each sequence stores \(L+1\) states and \(L\) regime labels. One label belongs to each state transition.

| Preset and split | Sequences | States array | Regimes array | Prediction windows |
|---|---:|---|---|---:|
| Full train, per seed | 256 | `(256, 193, 4)` | `(256, 192)` | 47,360 |
| Full validation, per seed | 64 | `(64, 193, 4)` | `(64, 192)` | 11,840 |
| Full test, per seed | 64 | `(64, 193, 4)` | `(64, 192)` | 11,840 |
| Smoke train | 32 | `(32, 97, 4)` | `(32, 96)` | 2,848 |
| Smoke validation | 8 | `(8, 97, 4)` | `(8, 96)` | 712 |
| Smoke test | 8 | `(8, 97, 4)` | `(8, 96)` | 712 |

The number of windows per sequence is \(L-C+1\): 185 for the full preset and 89 for the smoke preset. Across the three full seeds there are 142,080 training windows, but **each fitted model uses only its own seed's 47,360 windows**. The seeds are separate repetitions, not one pooled training dataset.

### 5.2 Independent random streams

For each experiment seed, `np.random.SeedSequence(seed).spawn(3)` supplies separate child random streams for train, validation, and test. The generator runs independently for each split. No single trajectory is cut into overlapping train and test portions.

Windows do overlap inside a split. That is intentional: adjacent next-step prediction examples naturally share history. It also means the 11,840 test windows are not 11,840 independent statistical replications. Treating every window as independent when calculating a confidence interval would overstate the amount of evidence.

### 5.3 Training-only normalization

For coordinate \(j\), the code computes a training-state mean \(\mu_j\) and standard deviation \(s_j\), pooling over all training sequences and all retained states, including their final states. It clips \(s_j\) below at \(10^{-6}\), then transforms

\[
\tilde x_{t,j}=(x_{t,j}-\mu_j)/s_j.
\]

The same training-derived statistics normalize training inputs, training targets, validation data, and test data. Validation and test states do not contribute to the normalization fit. Including target states from the **training** split is legitimate supervised preprocessing; it does not use held-out data.

Evaluation converts predictions back with \(\hat x_{t,j}=s_j\hat{\tilde x}_{t,j}+\mu_j\). Consequently, training/validation history files and reported test errors use different scales. Their raw numerical values should not be compared directly.

### 5.4 Exact window alignment

For every \(t\in\{C-1,\ldots,L-1\}\), the example is

\[
u_t=\operatorname{flatten}(\tilde x_{t-C+1},\ldots,\tilde x_t),
\qquad y_t=\tilde x_{t+1},
\qquad \text{label}=r_t.
\]

For \(C=8\), the first example uses states 0 through 7 to predict state 8; its label is regime 7. The last uses states 184 through 191 to predict state 192; its label is regime 191.

Flattening preserves chronological order: all four coordinates of the earliest state, then all four coordinates of the next state, and so on. A batch has input shape `(B, 32)`, target shape `(B, 4)`, and label shape `(B,)`.

The `TensorDataset` carries labels for every model's loader so one common pipeline can serve all models. However, the training and prediction code passes these labels into a model only when its name/mode is `oracle`. The presence of a label in the dataset does not mean the learned router receives it.

Inputs and targets are converted to float32. Regimes are int64. The generator's matrix array is float32 and saved states are float32; NumPy's random draws and intermediate simulation state can use higher precision before storage.

### 5.5 Dataset storage and provenance

Datasets are compressed NumPy archives under

```text
E:/HarunoKen/Twilight_MoE_Dynamics/
├── generated/<dataset fingerprint>/
│   ├── train.npz
│   ├── val.npz
│   └── test.npz
├── metadata/<dataset fingerprint>.json
├── raw/
├── processed/
└── test_runs/<unique temporary-test directory>/
```

`raw` and `processed` are reserved directories; the current workflow primarily uses `generated` and `metadata`. No external dataset is downloaded.

The dataset fingerprint is the first 12 hexadecimal characters of a SHA-256 hash over sorted JSON containing `DATA_VERSION`, the data configuration, and the seed. Metadata records those inputs, the transition matrices, regime alignment, and full SHA-256 checksums of the three archives.

When a complete cached dataset exists, `ensure_data` recomputes archive checksums before reusing it. A mismatch raises an error. If the metadata or a split is missing, the function regenerates the split set. Generator changes require manually incrementing `DATA_VERSION`; the dataset cache does not automatically hash generator source code.

Source: [data.py](D:/HarunoKen/Twilight_MoE_Dynamics/src/twilight_moe/data.py).

## 6. The five model families

### 6.1 Shared expert/dense MLP form

Every expert and dense baseline has the form

\[
f(u)=W_3\tanh\!\left(W_2\tanh(W_1u+b_1)+b_2\right)+b_3.
\]

Both hidden layers use the same width. The output has four coordinates and no output activation. There is no dropout, batch normalization, residual connection, or explicit linear-dynamics constraint. `nn.Linear` layers use their ordinary PyTorch initialization under the chosen random seed.

The networks operate on normalized input histories and predict normalized next states. They are trained directly for prediction, rather than to reconstruct an autoencoder input.

### 6.2 Dense active-parameter control

`dense_active` uses one network for every input. Its hidden width is chosen to approximately match the learned MoE's per-input active-parameter proxy: the complete router plus the selected experts.

In the full preset, its architecture is `32 → 57 → 57 → 4`. It has 5,419 parameters, all trainable and all active on each example. The target proxy is 5,355, so the mismatch is approximately +1.20%.

### 6.3 Dense total-parameter control

`dense_total` approximately matches all parameters stored in the learned MoE, including experts not selected for a particular input.

Its full-preset architecture is `32 → 70 → 70 → 4`, with 7,564 parameters. The learned MoE has 7,599, so the mismatch is approximately −0.46%.

The two controls address different questions. The first compares models with similar per-example parameter participation. The second asks whether having several experts helps beyond simply giving one dense model a similar total number of weights.

### 6.4 Learned MoE

The learned model has three independently parameterized expert MLPs, each `32 → 32 → 32 → 4`, and a router `32 → 24 → 3`. The router uses a `tanh` hidden layer and a final softmax.

For input \(u\), let router probabilities be \(p_0(u),p_1(u),p_2(u)\). The router selects the two largest probabilities, forming the set \(S(u)\). The selected weights are renormalized:

\[
w_e(u)=\frac{p_e(u)}{\sum_{j\in S(u)}p_j(u)},\quad e\in S(u),
\qquad \hat y(u)=\sum_{e\in S(u)}w_e(u)f_e(u).
\]

For example, probabilities `(0.60, 0.30, 0.10)` select experts 0 and 1 and use weights `(2/3, 1/3)`. The third expert is not evaluated for that input.

No expert is assigned a regime in advance. Learned expert 0 does not inherently mean true regime 0. Specialization, if it emerges, is an outcome of optimization.

### 6.5 Frozen random-router control

`random` uses the same three expert MLPs but replaces the learned router with a frozen bias-free linear projection `32 → 3`, followed by softmax and the same top-k rule.

The random weights are initialized once from the seed and have `requires_grad=False`. Routing is deterministic and input-dependent: repeating the same input produces the same selection and mixture weights. This is **not** a new random expert draw at every batch, and it is not input-independent uniform sampling.

The experts still train normally. This control asks how much the learned router contributes beyond a fixed partition and mixture of the input space. It is not a perfect isolation of router learning because the frozen router also differs in architecture and parameter count.

### 6.6 Regime-informed oracle

`oracle` has the same three expert MLPs but no neural router. For an example with label \(r_t\), it selects expert \(r_t\) with weight 1. Its effective top-k is always 1, regardless of the configuration's top-k value.

The oracle is privileged during both training and evaluation. Training examples from each regime update that regime's designated expert. Ordinary models must infer useful routing from the observed history instead.

The trained oracle does not receive the actual transition matrix. It still has to learn a predictor from data. Its extra information makes it a useful reference, but finite training, architecture, and distribution shift can prevent it from being a numerical upper bound on every performance metric.

### 6.7 Two references that are not trained neural models

**Persistence:** predict \(\hat x_{t+1}=x_t\). This checks whether a neural model improves on simply keeping the latest observed state unchanged.

**Known dynamics with true regime:** predict \(\hat x_{t+1}=A_{r_t}x_t\). This uses the generator's actual matrices and labels. Its one-step residual is exactly the process-noise term, up to numerical precision. Its expected coordinate-averaged MSE is \(\sigma^2=0.0064\).

The known-dynamics predictor is calculated as an evaluation reference. It is separate from the trained oracle network and is not included as a sixth trained model.

Sources: [models.py](D:/HarunoKen/Twilight_MoE_Dynamics/src/twilight_moe/models.py), [routers.py](D:/HarunoKen/Twilight_MoE_Dynamics/src/twilight_moe/routers.py), and [evaluation.py](D:/HarunoKen/Twilight_MoE_Dynamics/src/twilight_moe/evaluation.py).

## 7. Routing implementation and gradients

### 7.1 Sparse dispatch is implemented explicitly

The forward pass obtains selected expert indices of shape `(B, k)` and weights of the same shape. For each expert, `torch.where(indices == expert_id)` identifies only the rows assigned to that expert. The expert evaluates those rows. Its weighted outputs are accumulated with `index_add` into a `(B, 4)` prediction tensor.

There is no calculation of every expert output followed by multiplication by zero. Nevertheless, several or all experts may execute somewhere within a batch because different examples choose different experts. “Two active experts” describes one example, not necessarily the entire batch.

The implementation loops over experts in Python. It has no fused dispatch kernel, capacity factor, token dropping, expert parallelism, or distributed load-balancing infrastructure. Sparse parameter participation therefore does not imply a wall-clock speed advantage on these tiny CPU networks.

### 7.2 Why top-2 is the default

With two selected experts, changes in their relative gate weights alter the prediction, so the prediction loss can train the router through those weights. The integer selection operation itself is not differentiable. For a locally unchanged selected set, the renormalized top-2 weights are a softmax restricted to that set; the excluded logit has no direct prediction-loss gradient through that restricted mixture. Load balancing supplies an additional router gradient.

With hard top-1 routing and a selected weight fixed at 1, ordinary differentiation through the weight would not train routing from prediction error. The implementation therefore uses a straight-through estimator when `top_k=1`:

\[
w=1+p_{\mathrm{selected}}-\operatorname{stopgrad}(p_{\mathrm{selected}}).
\]

The forward value is 1, while the backward pass uses the selected probability's derivative. This is a surrogate gradient, not the exact derivative of discrete expert selection. The full saved experiment uses top-2; top-1 is supported and tested but not the reported preset.

### 7.3 Load-balancing term

For a batch of \(B\) examples and \(k\) selections each, define

\[
f_e=\frac{1}{Bk}\sum_{b=1}^{B}\sum_{s=1}^{k}\mathbf1[i_{b,s}=e],
\qquad q_e=\frac{1}{B}\sum_{b=1}^{B}p_e(u_b).
\]

The auxiliary term is

\[
\mathcal L_{\mathrm{balance}}=E\sum_{e=0}^{E-1}\operatorname{stopgrad}(f_e)q_e.
\]

`f_e` is the fraction of selected slots, not the fraction of examples that select expert \(e\). Its components sum to 1. The frequency term is detached; gradients flow through the average router probabilities.

The learned MoE optimizes prediction MSE plus `0.01 × balance`. Uniform slot usage makes this term equal to 1, rather than zero. Its absolute value is not itself a prediction error. The term discourages concentration but does not guarantee equal expert use or regime-aligned specialization.

The random and oracle controls do not add this auxiliary term to their loss. The random mode computes a balance value inside its shared forward implementation, but the training code does not use it.

## 8. Parameter accounting

For input dimension \(d=32\) and hidden width \(h\), the three-layer MLP has

\[
N_{\mathrm{MLP}}=(d+1)h+(h+1)h+(h+1)4=h^2+38h+4.
\]

At expert width 32, this is \(2244\) parameters per expert. The learned router has

\[
N_{\mathrm{router}}=(32+1)24+(24+1)3=867.
\]

The learned MoE therefore stores \(3(2244)+867=7599\) parameters and has a per-example active proxy of \(2(2244)+867=5355\).

| Full-preset model | Total stored parameters | Trainable parameters | Active parameter proxy |
|---|---:|---:|---:|
| Dense active | 5,419 | 5,419 | 5,419 |
| Dense total | 7,564 | 7,564 | 7,564 |
| Learned MoE | 7,599 | 7,599 | 5,355 |
| Random router | 6,828 | 6,732 | 4,584 |
| Oracle | 6,732 | 6,732 | 2,244 |

The random router's 96 frozen weights count toward total and active capacity, but not trainable capacity. The oracle has no router parameters.

Dense widths are found by exhaustive search over integer widths 1 through 4096, minimizing absolute parameter mismatch. Under the smoke configuration, the dense widths are 44 and 55; experts have width 24 and the learned router hidden width is 16.

These counts include biases and the full router. They exclude activation memory, optimizer state, indexing cost, softmax operations, communication, and hardware utilization. They are not FLOP counts. Training-time measurements in the result files include the training and validation loop and best-state restoration, but exclude data generation and subsequent test diagnostics. They are not controlled inference benchmarks.

## 9. Training procedure

### 9.1 Loss and optimizer

For normalized predictions and targets, the prediction loss is

\[
\mathcal L_{\mathrm{pred}}=\frac{1}{4B}\sum_{b=1}^{B}\sum_{j=1}^{4}
(\hat{\tilde x}_{b,j}-\tilde x_{b,j})^2.
\]

Dense, random, and oracle models minimize this loss. The learned MoE minimizes

\[
\mathcal L=\mathcal L_{\mathrm{pred}}+0.01\mathcal L_{\mathrm{balance}}.
\]

The optimizer is Adam. The full preset explicitly sets learning rate 0.001 and otherwise uses PyTorch's default Adam settings. No weight decay, learning-rate schedule, mixed precision, or explicit optimizer customization is configured. Parameters with `requires_grad=False` are excluded from the optimizer.

Before each optimizer step, gradients are clipped to global norm 5. Nonfinite total training loss raises an exception. The code does not provide an automatic recovery/retry mechanism for such a failure.

### 9.2 Reproducibility setup

Before each model is constructed, `seed_everything` resets Python's random seed, NumPy's legacy global seed, and PyTorch's seed. It sets PyTorch CPU threads to 2 and enables deterministic algorithms. Data generation separately uses the seeded NumPy `Generator` streams described earlier.

The shuffled training `DataLoader` has its own PyTorch generator initialized with the experiment seed. Models within the same seed therefore use the same dataset and seeded batch ordering. Different architecture shapes consume different numbers of initialization draws, so identical seed numbers do not generally imply identical weights across architectures. One useful exception is the three MoE-family modes: each constructs its identically shaped experts before constructing any router, so their initial expert weights match within a seed. Their gate construction and subsequent training then differ.

The code currently runs on CPU. It does not move models or tensors to CUDA. Notebook 00 reports CUDA availability only as environment information.

### 9.3 Epoch loop

```python
seed_everything(seed)
model = build_model(model_name, config)
optimizer = Adam(trainable_parameters, lr=learning_rate)

for epoch in range(1, epochs + 1):
    model.train()
    for inputs, targets, regimes in shuffled_training_loader:
        optimizer.zero_grad()
        prediction, balance, routes = model(
            inputs, regimes if model_name == "oracle" else None
        )
        loss = prediction_mse(prediction, targets)
        if model_name == "moe":
            loss += balance_weight * balance
        loss.backward()
        clip_global_gradient_norm(model, 5.0)
        optimizer.step()

    model.eval()
    validation_mse = evaluate_normalized_validation_mse_without_gradients()
    if validation_mse < best_validation_mse:
        save_deep_copy_of_parameters()

restore_best_parameters()
```

This is explanatory pseudocode. The actual implementation is in `training.py`.

The full training set gives 93 batches per epoch: 92 batches of 512 and one batch of 256. All windows are used; the last batch is not dropped. At 40 epochs this is 3,720 optimizer steps per fitted model. Validation uses batches of up to 4,096 windows and no gradient computation.

### 9.4 Checkpoint selection

The program always runs all configured epochs. There is no early stopping. It keeps a deep copy whenever normalized validation MSE strictly improves, then restores the best copy after training. Equal validation scores retain the earlier checkpoint. The auxiliary balancing term is not part of checkpoint selection.

Because normalization is coordinate-specific, selecting normalized MSE is equivalent to weighting original-coordinate squared errors by \(1/s_j^2\). The reported original-unit test MSE is an unweighted coordinate average. These criteria are close in the isotropic dataset but are not mathematically identical under arbitrary unequal coordinate variances.

`train_mse_normalized` records prediction MSE averaged across batches, not total loss including load balancing. The training MSE is accumulated while parameters change during the epoch; validation MSE evaluates the end-of-epoch model. Neither router entropy, auxiliary loss, nor gradient norm is saved in the training-history CSV.

The full experiment performs 15 fits: five model families times three seeds. Seeds 7, 17, and 27 vary both data realization and model initialization, so the observed seed spread combines those sources of variability.

Source: [training.py](D:/HarunoKen/Twilight_MoE_Dynamics/src/twilight_moe/training.py).

## 10. One-step evaluation

After selecting a model, the program predicts the next state from every eligible **observed** test history. Predicted states are not fed into later histories for this metric.

For sequence \(n\) and target transition \(t\), define

\[
e_{n,t}=\frac14\sum_{j=1}^{4}(\hat x_{n,t+1,j}-x_{n,t+1,j})^2.
\]

`test_mse` averages \(e_{n,t}\) across all eligible windows. `test_error` saves the per-window values, with shape `(64, 185)` in the full preset. Each window contributes equally; with equal sequence lengths this also gives equal total weight to each test sequence.

`switch_mse` averages only windows whose current regime age is zero. `regime_ages` starts at zero at sequence start, increments when the label is unchanged, and resets on a change. Because valid prediction windows start at time 7, the initial artificial age-zero value at time 0 does not enter switch MSE in the supplied presets.

`persistence_mse`, `known_dynamics_mse`, and `noise_variance` provide the reference values described above. The finite-sample known-dynamics MSE need not equal the configured noise variance exactly.

## 11. Routing diagnostics

### 11.1 Expert utilization

Utilization counts every selected expert slot over all test windows, then divides by the total number of slots. With top-2 and 11,840 windows, the denominator is 23,680. It is a selection-frequency measure, not the total mixture weight received by each expert.

For example, selecting an expert with weight 0.01 counts the same as selecting it with weight 0.99. With two distinct selected experts per example, an individual expert's slot share cannot exceed one-half. The oracle uses one selected slot and therefore has a different utilization denominator.

### 11.2 Expert–regime agreement

For each window, the diagnostic takes the highest-probability selected expert, not both selected experts. Because expert IDs are arbitrary, it enumerates the six permutations of three labels and chooses the mapping with highest agreement on **validation** data. That fixed mapping is then applied to the test routes.

The reported agreement is

\[
\frac1{N_{\mathrm{test}}}\sum_i
\mathbf1[\pi(e_i^{\mathrm{top}})=r_i].
\]

This prevents selecting the most favorable expert-label mapping directly on the test set. It does not make routing accuracy a causal interpretability test. A model can predict well without one expert per regime, and top-2 mixture behavior is only partly described by its top expert.

The saved confusion matrix uses true regime on rows and **raw expert IDs** on columns. It is not relabeled by the optimal permutation. For the oracle, agreement is 1 by construction and should not be presented as learned discovery.

## 12. Adaptation after a switch

### 12.1 Event-aligned error

`event_aligned` finds actual label changes at times \(t\geq C-1\). Lag zero is the prediction made with the newly selected regime's first transition. For each lag, it averages errors over switch episodes that survive until that lag without another switch or sequence end.

The full preset plots lags 0 through 23. It also saves the number of contributing episodes at every lag. Falling support matters: late points describe a subset of longer episodes rather than the exact same episode population as lag zero.

These errors use observed histories, just like one-step evaluation. They do not measure free-running rollout error, and there is no online gradient update after a switch. “Adaptation” here means the fixed model responds to new observations in its moving history window.

### 12.2 Validation-derived recovery threshold

For each model, validation windows with regime age at least the context length, 8, are called late-regime windows. The threshold is

\[
\tau_m=1.5\times\operatorname{mean}\{e^{\mathrm{val}}_{m,n,t}:\operatorname{age}_{n,t}\geq8\}.
\]

For each eligible test switch, recovery occurs when three consecutive window errors are at or below \(\tau_m\), before the next switch or sequence end. The recorded delay is the **start** of the first successful three-step run.

If errors at lags 2, 3, and 4 are all below threshold and no earlier three-step run qualifies, delay is 2. It could only be confirmed retrospectively at lag 4. It is therefore not the timestamp of an online detection decision.

The output includes the total eligible episode count, recovered episode count, recovered fraction, mean delay among recovered episodes, and the threshold. Unrecovered episodes do not contribute a fabricated delay of zero. They are excluded from the delay mean and retained in the fraction's denominator. Episodes shorter than three available predictions cannot satisfy the criterion.

No survival-analysis estimator is fitted. “Censored” describes why some delays remain unavailable; the code does not estimate their eventual recovery times.

### 12.3 How to compare this metric

Thresholds differ by model because they use each model's own validation error. A weaker model can receive a larger, easier threshold. Delays also condition on recovery and are affected by noise and episode duration. Consequently, recovery delay must be read alongside recovered fraction, absolute MSE, and the event-aligned curve. It is not a standalone ranking of regime-identification ability.

## 13. Multi-step rollout evaluation

One-step evaluation repeatedly supplies actual observed histories. Rollout evaluation instead starts from an observed history and recursively feeds predictions back:

\[
\hat x_{t+1}=f(x_{t-C+1:t}),\qquad
\hat x_{t+2}=f(x_{t-C+2:t},\hat x_{t+1}),\quad\ldots
\]

The implementation keeps the rolling history in normalized coordinates, drops its oldest state, and appends the new normalized prediction. Original-unit predictions are used for scoring. No future observed state is injected after rollout initialization. No future noise is sampled by the neural models; they produce deterministic point forecasts.

Full-preset origins are times 7, 39, 71, 103, and 135. Each is evaluated for 32 steps, giving five origins per sequence and 320 origins per test seed. The final partial horizon is omitted. Target intervals do not overlap within a sequence, although contexts may overlap other parts of that same sequence and all origins from a sequence remain statistically related.

The oracle receives the true future regime at every rollout step. This is privileged future information. Ordinary models route from their evolving, partly or entirely predicted histories.

`rollout_mse[h-1]` averages coordinate errors across origins at horizon \(h\). `rollout_mean_mse` averages these horizon-specific MSE values equally over horizons 1–32. The persistence rollout holds the original last observed state fixed at every horizon; it does not update to future observations.

The horizon-1 rollout MSE can differ from `test_mse`, because it uses only 320 selected origins rather than all 11,840 eligible one-step windows.

A trained oracle can be best at one step and worse at long horizons. Each oracle expert is an unconstrained nonlinear history-based MLP, not the exact stable matrix. Repeated predictions can drift away from its training histories. The experiment observes the error pattern; it does not contain an intervention that proves which mechanism causes it.

Source for Sections 10–13: [evaluation.py](D:/HarunoKen/Twilight_MoE_Dynamics/src/twilight_moe/evaluation.py).

## 14. Configuration reference

| Setting | Smoke | Full experiment | Role |
|---|---:|---:|---|
| Seeds | `[7]` | `[7,17,27]` | Data and model repetitions |
| Train sequences | 32 | 256 | Independent training trajectories |
| Validation sequences | 8 | 64 | Checkpoint selection and diagnostics |
| Test sequences | 8 | 64 | Held-out evaluation |
| Transitions per sequence | 96 | 192 | Retained simulation length |
| Context states | 8 | 8 | Input history length |
| Burn-in transitions | 64 | 128 | Discarded initial steps |
| Noise standard deviation | 0.08 | 0.08 | Per-coordinate process noise |
| Stay probability | 0.95 | 0.95 | Regime persistence |
| Experts | 3 | 3 | Expert count, fixed to three in version 1 |
| Expert hidden width | 24 | 32 | Both expert hidden layers |
| Learned router hidden width | 16 | 24 | Router hidden layer |
| Top-k | 2 | 2 | Selected experts, excluding oracle override |
| Epochs | 6 | 40 | Fixed training duration |
| Batch size | 256 | 512 | Training windows per batch |
| Learning rate | 0.002 | 0.001 | Adam learning rate |
| Balance weight | 0.01 | 0.01 | Learned-router auxiliary loss weight |
| CPU threads | 2 | 2 | PyTorch thread count |
| Rollout horizon | 16 | 32 | Recursive prediction length |
| Adaptation window | 20 | 24 | Number of plotted switch lags |
| Recovery multiplier | 1.5 | 1.5 | Multiplier on late-validation MSE |
| Consecutive recovery steps | 3 | 3 | Required below-threshold run |

Smoke and full differ in several parameters, so their result difference cannot be attributed solely to training longer. The smoke run is a functional check, not a controlled sample-size ablation.

`load_config` resolves an explicit path first, then `TWILIGHT_CONFIG`, then `configs/smoke.yaml`. A relative configuration path is resolved against the project root. `TWILIGHT_DATA_DIR`, when set, overrides the YAML dataset root.

Validation asserts three experts, legal top-k, a context shorter than the sequence, positive split counts, positive epochs/batch size, noise and switching constraints, and a feasible rollout horizon. This is not a comprehensive configuration schema: some settings have no explicit range/type checks, and Python assertions can be disabled with optimized execution. Do not assume arbitrary YAML edits are supported.

Sources: [config.py](D:/HarunoKen/Twilight_MoE_Dynamics/src/twilight_moe/config.py), [smoke.yaml](D:/HarunoKen/Twilight_MoE_Dynamics/configs/smoke.yaml), and [experiment.yaml](D:/HarunoKen/Twilight_MoE_Dynamics/configs/experiment.yaml).

## 15. Results of the completed full experiment

### 15.1 One-step prediction

All entries below are original-unit, coordinate-averaged MSE. The SD column is the sample standard deviation across the three seeds, not a standard error or confidence interval.

| Model | Seed 7 | Seed 17 | Seed 27 | Mean | Sample SD |
|---|---:|---:|---:|---:|---:|
| Dense active | 0.011182 | 0.011312 | 0.010898 | 0.011131 | 0.000212 |
| Dense total | 0.011116 | 0.011202 | 0.010884 | 0.011068 | 0.000165 |
| Learned MoE | 0.011591 | 0.011479 | 0.011328 | 0.011466 | 0.000132 |
| Random router | 0.011750 | 0.011870 | 0.011494 | 0.011705 | 0.000192 |
| Oracle | 0.006621 | 0.006709 | 0.006682 | 0.006671 | 0.000045 |

Persistence averaged 0.039325. The known-dynamics reference averaged 0.006396, close to the configured noise variance of 0.006400. All five trained families improved on persistence in this experiment.

![One-step comparison for the full experiment](D:/HarunoKen/Twilight_MoE_Dynamics/results/figures/experiment_57adae732ea0/one_step_comparison.png)

For each seed \(s\), the paired difference against a dense control is \(\Delta_s=MSE_{\mathrm{MoE},s}-MSE_{\mathrm{dense},s}\). Positive values favor the dense control.

| Comparison | Mean paired MSE difference | SD of paired differences | Mean within-seed relative change | Seeds favoring MoE |
|---|---:|---:|---:|---:|
| MoE minus dense active | +0.000335 | 0.000146 | +3.03% | 0/3 |
| MoE minus dense total | +0.000399 | 0.000106 | +3.61% | 0/3 |

The relative change is the mean of \(100(MSE_{\mathrm{MoE},s}/MSE_{\mathrm{dense},s}-1)\), not a percentage calculated from rounded table entries.

The learned router outperformed the frozen random router on one-step MSE in all three seeds. Thus there is evidence of a useful learned-routing contribution relative to this random control, while the overall architecture still trails the dense controls. These are distinct comparisons.

### 15.2 What the oracle comparison establishes

The trained oracle's mean MSE of 0.006671 is only modestly above the known-dynamics reference. Giving experts the correct regime is helpful, and the expert architecture can approximate the regime-conditioned one-step maps reasonably well.

It does not follow that all of the ordinary models' gap is an optimization failure of the router. Ordinary models lack current-regime information at an unpredictable switch, while the oracle receives it. Some additional error is intrinsic to the information available to a non-oracle predictor. The experiment does not calculate the exact Bayes-optimal error given only an eight-state history.

### 15.3 Checkpoints and learning curves

| Model | Selected epoch, seed 7 | Seed 17 | Seed 27 |
|---|---:|---:|---:|
| Dense active | 39 | 38 | 39 |
| Dense total | 37 | 38 | 38 |
| Learned MoE | 37 | 40 | 39 |
| Random router | 40 | 40 | 40 |
| Oracle | 39 | 33 | 35 |

![MoE-family validation learning curves for seed 7](D:/HarunoKen/Twilight_MoE_Dynamics/results/figures/experiment_57adae732ea0/learning_moe.png)

Most selected epochs are near the training limit. This supports examining training duration in a future validation-controlled study; it does not demonstrate that a longer run would reverse the ranking. No training-length sweep or equal-budget hyperparameter search has been performed.

### 15.4 Routing behavior

| Model | Test routing agreement, seed 7 | Seed 17 | Seed 27 | Mean |
|---|---:|---:|---:|---:|
| Learned MoE | 45.82% | 60.08% | 48.28% | 51.39% |
| Random router | 35.52% | 34.47% | 36.43% | 35.47% |
| Oracle | 100% | 100% | 100% | 100% |

The learned router is more regime-correlated than this frozen random control, but its experts do not form a perfect regime classifier. One-third is only a rough balanced-class reference; finite class imbalance and the selected validation mapping make it inappropriate to treat that reference as a formal significance threshold.

![Expert use for the first full-run seed](D:/HarunoKen/Twilight_MoE_Dynamics/results/figures/experiment_57adae732ea0/expert_utilization.png)

For seed 7, the learned MoE's selected-slot shares are 42.12%, 26.82%, and 31.05%. All experts are used. This is not complete collapse to a single selected pair, but neither is it perfectly balanced use. It also says nothing by itself about how different the learned expert functions are.

![True-regime versus raw-expert counts for learned routing, seed 7](D:/HarunoKen/Twilight_MoE_Dynamics/results/figures/experiment_57adae732ea0/routing_confusion.png)

The matrix shows mixed assignment rather than a clean one-expert/one-regime diagonal. Since this is top-2 routing, the secondary expert and the continuous mixture weights may carry useful information omitted by the top-expert diagnostic.

### 15.5 Switch error and recovery

The following entries are arithmetic means of the three seed-level metrics. The recovery-delay mean is therefore a mean of per-seed conditional means, not one pooled episode-weighted estimate.

| Model | Switch-window MSE | Recovered episodes, mean fraction | Mean delay among recovered episodes, steps |
|---|---:|---:|---:|
| Dense active | 0.070248 | 76.42% | 2.652 |
| Dense total | 0.070517 | 76.24% | 2.652 |
| Learned MoE | 0.069203 | 76.17% | 2.920 |
| Random router | 0.069041 | 75.59% | 2.936 |
| Oracle | 0.006916 | 81.62% | 1.480 |

The three seeds have 578, 556, and 580 eligible switch episodes, respectively. The very large non-oracle lag-zero errors are consistent with the causal timing of hidden switches. The ordinary models' average switch-window ranking differs slightly from their overall one-step ranking; a single subset metric should not replace the primary comparison.

![Switch-aligned error and number of contributing episodes, seed 7](D:/HarunoKen/Twilight_MoE_Dynamics/results/figures/experiment_57adae732ea0/switch_adaptation.png)

Even the oracle can have a positive recovery delay under this definition. Independent noise can keep an error above threshold, and the three-step condition is stricter than one correct regime selection. Thus a positive oracle delay is not evidence that its routing adapts slowly.

### 15.6 Free-running rollouts

| Model | Mean over horizons 1–32 | Sample SD across seeds | Horizon-32 MSE, seed mean |
|---|---:|---:|---:|
| Dense active | 0.057350 | 0.000529 | 0.064569 |
| Dense total | 0.057107 | 0.000901 | 0.064628 |
| Learned MoE | 0.057571 | 0.002021 | 0.064082 |
| Random router | 0.057142 | 0.004668 | 0.062838 |
| Oracle | 0.054049 | 0.002282 | 0.077645 |

![Free-running prediction error across horizons](D:/HarunoKen/Twilight_MoE_Dynamics/results/figures/experiment_57adae732ea0/rollout_error.png)

The trained oracle is best on average across the 32 horizons, helped by its early-horizon advantage, but it is worse at horizon 32 than the other trained models. A known exact-matrix rollout benchmark is not implemented, so this graph must not be labeled an exact oracle bound.

The ordinary models have close mean rollout scores and noticeable seed variation. The learned MoE does not show a consistent advantage on this metric. The random control's lower horizon-32 mean does not establish superior dynamical understanding; shrinking predictions toward a central value can sometimes reduce long-horizon squared error in a noisy stable system.

Theoretical stationary variance, about 0.054983 per coordinate, helps orient the scale of long-horizon errors. It is not a measured zero-predictor benchmark in the program, and the finite set of selected rollout targets need not have exactly that empirical variance.

### 15.7 What remains unexplained

The current results do not identify a single cause for the learned MoE's deficit. Plausible hypotheses include a task simple enough for a dense network, additional router optimization difficulty, the top-2 mixture constraint, balancing-loss tradeoffs, and insufficient tuning. Those are candidate explanations, not demonstrated causes.

The ordinary predictor's conditional mean has the conceptual form

\[
\mathbb E[x_{t+1}\mid x_{t-C+1:t}]
=\sum_r \Pr(r_t=r\mid x_{t-C+1:t})A_rx_t.
\]

This gives a reason a mixture could be useful, but it is not the architecture's exact implemented parameterization: its experts are unrestricted nonlinear history-based networks. Nor does a conceptual mixture form establish that sparse top-2 routing will beat a dense approximation at this scale.

Primary numerical source: [full-run summary.csv](D:/HarunoKen/Twilight_MoE_Dynamics/results/metrics/experiment_57adae732ea0/summary.csv), supplemented by the per-model diagnostic archives in that same run directory.

## 16. Notebook-by-notebook guide

The notebooks expose configuration, loading, training calls, calculations, and plotting in separate cells. Substantial algorithms live in importable source modules rather than being duplicated in each notebook. “Run a cell” therefore sometimes calls a documented reusable function; it does not mean every training-loop line is copied into notebook cells.

| Notebook | What to inspect or run | Outputs and interpretation |
|---|---|---|
| `00_environment_check` | Interpreter, packages, CPU/CUDA report, configuration, paths | Confirms which environment and preset you are using |
| `01_generate_switching_data` | Matrix norms, Markov matrix, independent data generation, normalization | Dataset paths, split sizes, training normalization statistics |
| `02_inspect_and_visualize_data` | Training-only trajectory and regime coverage | `training_trajectory.png` and `regime_coverage.png`; checks data appearance and coverage |
| `03_train_dense_baselines` | Model structures and counts, training both dense controls, learning curves | Dense checkpoints, metrics, histories, `learning_dense_active.png` |
| `04_train_small_moe` | Learned/random/oracle model construction and training | Routing-family checkpoints, metrics, histories, `learning_moe.png` |
| `05_compare_performance` | Completeness check, one-step comparison, paired differences, capacity mismatch | `one_step_comparison.png`, exact tables, saved-run narrative |
| `06_analyze_routing` | Agreement, slot use, confusion, switch-aligned error, recovery table | `expert_utilization.png`, `routing_confusion.png`, `switch_adaptation.png` |
| `07_multi_step_rollout` | Load recursive forecast diagnostics and compare horizons | `rollout_error.png`, rollout average table |

Nine figures are produced. Each has editable plotting code and appears beneath its notebook cell, as well as being saved as a PNG. The shared plotting module sets sizes, fonts, restrained grids, and fixed colors for model identities; figure saving uses 160 DPI and a tight bounding box.

Data visualization, learning curves, utilization, confusion, and switch curves show the **first configured seed**. One-step comparison and rollout comparison incorporate all configured seeds. In the rollout chart, light individual lines show seed variation, while stronger lines show means. These light lines are not confidence bands.

Notebook 05's descriptive summary is a saved-run markdown cell. Its text does not automatically recalculate when one executes a different preset manually. The automated notebook runner removes the old summary before execution; `summarize_results.py` creates a new one afterward. Always check the printed preset and computed tables before interpreting a saved narrative.

## 17. File and function reference

### 17.1 Source modules

| File | Functions/classes | Responsibility |
|---|---|---|
| `__init__.py` | `__version__` | Package version 0.1.0 |
| `config.py` | `fingerprint`, `load_config`, `run_dir`, `checkpoint_dir` | Resolve paths, validate basic settings, derive run identities |
| `dynamics.py` | `transition_matrices`, `generate_sequences` | Construct the fixed laws and simulate switched trajectories |
| `data.py` | `dataset_path`, `ensure_data`, `load_data`, `windows`, `regime_ages` | Cache/checksum data, fit normalization, build examples and regime ages |
| `routers.py` | `Router`, `sparse_gates` | Learned/frozen gate probabilities and top-k selection weights |
| `models.py` | `mlp`, `nparameters`, `Dense`, `MoE`, `build_model` | Define architectures, sparse forward pass, width matching, capacity |
| `training.py` | `seed_everything`, `fit` | Deterministic initialization, optimizer loop, validation checkpoint selection |
| `evaluation.py` | `predict`, `switch_recovery`, `event_aligned`, `routing_statistics`, `rollout`, `evaluate` | All predictive and diagnostic metrics |
| `experiment.py` | `run_models`, `source_hashes`, `load_metrics`, `load_diagnostics`, `load_checkpoint` | Coordinate fitting/evaluation and serialize results |
| `plotting.py` | `style`, `save_figure`, `COLORS` | Shared chart appearance and figure output paths |

`Dense.forward` and `MoE.forward` share a three-item return interface: prediction, auxiliary balance scalar, and selected expert indices. Dense models return zero and `None` for the last two items. This allows one training loop and one evaluation pipeline to handle all families.

`run_models` loops over seeds and model names, loads each seed's data, fits the requested models, evaluates the selected checkpoints, and writes their artifacts. It prints a short result line per fit and returns a DataFrame of the fits performed by that call. It then reconstructs `summary.csv` from all existing result JSON files for the configuration's model/seed combinations.

### 17.2 Scripts

| Script | Behavior and cautions |
|---|---|
| `run_experiment.py` | CLI entry point; accepts `--config`, defaults to smoke, runs all model families |
| `build_notebooks.py` | Generates all eight notebook sources with `nbformat`; overwrites their current contents and outputs, so avoid it after manual notebook edits unless regeneration is intended |
| `execute_notebooks.py` | Executes notebooks in filename order using fresh kernels; explicitly uses the current environment's Python, saves outputs, and exports HTML previews |
| `summarize_results.py` | Reads results, writes the experiment log, replaces notebook 05's saved narrative, and refreshes its HTML preview; it does not retrain models |
| `start_jupyter.ps1` | Changes to the project root and opens notebook 00 using `.venv`; it inherits any preset environment variable already set |

The automated notebook runner sets Jupyter, IPython, and Matplotlib runtime directories under `.runtime`, sets a 60-second kernel startup timeout and a 1,800-second cell timeout, validates notebook format, and shuts each kernel down after use. The saved notebook metadata records the executed configuration and completion status. HTML previews are static output views, not live computational kernels.

### 17.3 Project configuration and documentation

`pyproject.toml` defines the installable package, Python version requirement, dependency source, and pytest source/test paths. `requirements.txt` gives minimum dependency versions. `requirements-lock.txt` records the installed versions from the verified environment. `environment.yml` offers a Python 3.12 conda environment that installs the pip requirements.

`AGENTS.md` preserves development constraints for future work. `research_plan.md` states the hypotheses and deferred work. `README.md` is the starting guide. `VERIFICATION.md` records execution and inspection. `results/experiment_log.md` summarizes the last explicitly summarized run. This report explains the program in greater detail.

## 18. Result files and their exact meaning

Run identities are formed from the configuration name plus a 12-character SHA-256 prefix of the complete loaded configuration. This includes the resolved dataset-root string. The saved full run is `experiment_57adae732ea0`; the smoke run is `smoke_f8716cb6cb70`.

```text
D:/HarunoKen/Twilight_MoE_Dynamics/results/
├── metrics/<run identity>/
│   ├── config.json
│   ├── environment.json
│   ├── summary.csv
│   ├── <model>_seed<seed>.json
│   ├── <model>_seed<seed>_history.csv
│   └── <model>_seed<seed>_diagnostics.npz
├── checkpoints/<run identity>/<model>_seed<seed>.pt
├── figures/<run identity>/<figure>.png
├── previews/<notebook>.html
└── experiment_log.md
```

### 18.1 Metric JSON/CSV fields

| Field | Meaning |
|---|---|
| `model`, `seed`, `dataset` | Model family, experiment seed, dataset directory |
| `total_parameters` | All stored model parameters, including frozen router weights |
| `trainable_parameters` | Parameters eligible for optimization |
| `active_parameters` | Per-example router-plus-selected-expert parameter proxy |
| `best_epoch` | Epoch whose normalized validation MSE was lowest |
| `best_val_mse_normalized` | That selection score, without balance penalty |
| `training_seconds` | Wall time around the fit loop and best-state restoration |
| `test_mse`, `test_windows` | Original-unit average one-step error and its window count |
| `switch_mse` | Mean one-step error at age-zero switch windows |
| `persistence_mse` | One-step last-state predictor error |
| `known_dynamics_mse` | One-step exact-matrix, true-regime predictor error |
| `noise_variance` | Configured \(\sigma^2\), not a fitted estimate |
| `switch_episodes` | Eligible test switch episodes |
| `recovered_episodes`, `recovered_fraction` | Episodes meeting the consecutive-step criterion and their fraction |
| `recovery_delay_steps` | Conditional mean of successful recovery-run start lags |
| `recovery_threshold` | Model-specific original-unit threshold fitted on validation errors |
| `rollout_mean_mse`, `rollout_origins` | Mean across rollout horizons and number of initial histories |
| `routing_agreement` | Test top-expert agreement after validation-fitted permutation; absent for dense models |
| `route_mapping_validation` | The chosen expert-ID-to-regime permutation; absent for dense models |

Undefined diagnostic quantities can be NaN. The current JSON writer permits Python's `NaN` representation, which is not strict interoperable JSON; external consumers may need conversion to `null`. Dense models' absent routing fields become missing values when records are combined in a DataFrame.

### 18.2 Diagnostic arrays

For a full-run model, `test_error` has shape `(64,185)`, `adaptation_mse` and `adaptation_counts` have length 24, and `rollout_mse` and `rollout_persistence` have length 32. Routing models additionally save a `(3,3)` confusion matrix, a length-3 utilization vector, and `selected_experts` with shape `(64,185,2)` for learned/random or `(64,185,1)` for oracle.

Full per-window predicted state vectors and full rollout trajectories are not saved. They can be recomputed from checkpoints and datasets. Router probabilities and continuous mixture weights are also not included in the diagnostic archive.

### 18.3 Checkpoints and reproducibility limits

Each `.pt` file contains the selected model `state_dict`, family name, seed, full configuration, and normalization statistics converted to lists. `load_checkpoint` rebuilds the model, loads the weights using `weights_only=True`, restores normalization arrays, and switches to evaluation mode.

Optimizer state, batch position, and random-number-generator state are not saved. These are inference/re-evaluation checkpoints, not exact interrupted-training-resume checkpoints.

`environment.json` records Python, platform, PyTorch, NumPy, CPU execution, and SHA-256 hashes of the package's Python source files. It does not itself contain every dependency version; the lock file supplies the broader package inventory.

There are several practical cache limitations. Repeating a model/seed/configuration fit overwrites its artifacts. Run identity does not include code hashes. Source edits followed by a partial rerun can therefore leave a mixture of old and new model results under one configuration directory; the loader does not reject this automatically. The environment manifest is also rewritten by `run_models`. Rerun all families after algorithm changes and compare provenance before making a new comparison. Notebook previews and the experiment log are shared current views, while metric/checkpoint/figure directories are configuration-specific.

## 19. Tests and what they establish

The recorded test run passed 12 cases. There are ten named test functions; routing gradients/dispatch are parametrized over top-k values 1, 2, and 3, producing the additional cases.

| Test | Property checked |
|---|---|
| Stability and reproducibility | Matrix norm below one, repeatable generated arrays, correct shapes, finite states |
| Transition residual | Empirical residual mean and variance agree with configured process noise when aligned with the saved regime |
| Causal windows | Exact input/target/regime indexing on a small constructed sequence |
| Splits and normalization | Different generated split sequences, training-only mean, checksum corruption detection |
| Router gradients and sparse dispatch, three cases | Correct output/index shapes, exactly B×k expert-row evaluations, nonzero prediction-loss router gradient |
| Controls and capacity | Frozen random router, repeatable routes, oracle label requirement, correct oracle indices, approximately matched dense capacity |
| Recovery censoring | Known example yields the expected episode count, recovered fraction, and conditional delay |
| Route permutation | A known permutation can be selected on validation and applied successfully |
| Deterministic fitting | Repeated short fits have identical selected weights and the recorded best score equals the minimum validation history |
| Rollout feedback | A deterministic increment model produces the expected recursive trajectory and persistence errors |

Tests establish specific implementation properties. They do not prove the experiment is optimally tuned, the MoE is interpretable, all possible configuration values work, or performance generalizes beyond the synthetic setting. In particular, the permutation test is a small correctness case rather than an exhaustive adversarial leakage test, and capacity matching tests use the supplied preset rather than every possible width.

The previous execution record also reports that all eight notebooks ran under both presets, the full saved notebooks contain 34 executed code cells and nine figures, and all 15 full-run checkpoints reproduced their saved one-step MSE when independently reloaded. These are recorded build checks. During this report review, the implementation hashes were checked again against the saved full-run manifest and matched.

Sources: [test_experiment.py](D:/HarunoKen/Twilight_MoE_Dynamics/tests/test_experiment.py), [conftest.py](D:/HarunoKen/Twilight_MoE_Dynamics/tests/conftest.py), and [VERIFICATION.md](D:/HarunoKen/Twilight_MoE_Dynamics/VERIFICATION.md).

## 20. How to run and inspect the project

### 20.1 Open the existing environment

In PowerShell:

```powershell
Set-Location 'D:\HarunoKen\Twilight_MoE_Dynamics'
.\.venv\Scripts\python.exe -m jupyter lab notebooks/00_environment_check.ipynb
```

The environment is already installed on this computer. Run notebooks 00 through 07 in order when reproducing everything. Later analysis notebooks can load existing compatible metrics without retraining, provided their configuration matches the result directory.

### 20.2 Choose the full preset explicitly

The default remains smoke even though saved notebook outputs currently show the full run. Before launching Jupyter:

```powershell
$env:TWILIGHT_CONFIG = 'configs/experiment.yaml'
.\.venv\Scripts\python.exe -m jupyter lab notebooks/00_environment_check.ipynb
```

Or, inside a notebook before `cfg = load_config()`:

```python
os.environ['TWILIGHT_CONFIG'] = 'configs/experiment.yaml'
cfg = load_config()
```

After changing configuration, restart the kernel and rerun dependent cells so old in-memory data/models do not mix with new settings. For exploration, use a new YAML filename and meaningful `name`; preserve the original preset for comparison.

### 20.3 Run without interactive notebooks

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/run_experiment.py --config configs/smoke.yaml
.\.venv\Scripts\python.exe scripts/run_experiment.py --config configs/experiment.yaml
```

The CLI training script computes diagnostics and saves their arrays, but does not create the notebook figures. To execute the full notebook workflow, including plotting:

```powershell
.\.venv\Scripts\python.exe scripts/execute_notebooks.py --config configs/experiment.yaml
.\.venv\Scripts\python.exe scripts/summarize_results.py --config configs/experiment.yaml
```

Executing all notebooks includes training again in notebooks 03 and 04; it is not a plot-only command. Jupyter uses local kernel sockets. The original build required authorized execution outside the restricted agent sandbox because those sockets were blocked there; normal local Jupyter execution was successful.

### 20.4 Inspect a trained model without retraining

After the notebook setup cell has made the package importable:

```python
from twilight_moe.config import load_config
from twilight_moe.data import load_data
from twilight_moe.experiment import load_checkpoint
from twilight_moe.evaluation import predict

cfg = load_config('configs/experiment.yaml')
model, normalization = load_checkpoint(cfg, 'moe', seed=7)
data, _ = load_data(cfg, seed=7)
errors, routes = predict(model, data['test'], cfg['data']['context'], normalization)
print(errors.mean())       # about 0.011591 for this saved model
print(routes.shape)        # (64, 185, 2)
```

This loads the selected model, rather than the final training epoch if it was not selected. `predict` returns per-window errors and selected indices, not the raw predicted-state array.

### 20.5 What to change first for learning

To understand the data, inspect a trajectory and its regime panel in notebook 02. To understand capacity matching, print `build_model` outputs in notebooks 03 and 04 and reproduce the count formula. To understand routing, read `MoE.forward` alongside `sparse_gates`. To understand evaluation, start with `windows` and `predict`, then compare those paths with `rollout`.

Do not interpret an edited configuration's results before rerunning the required training/evaluation stages. Do not use `build_notebooks.py` as a normal launch command: it regenerates notebook source and can overwrite notebook edits.

## 21. Scientific scope, limitations, and useful next experiments

The present evidence supports a narrow statement: under one stable four-dimensional switching generator, one fixed training preset, and three combined data/initialization seeds, the learned top-2 MoE has slightly worse one-step MSE than both dense controls. It beats the particular frozen random router, uses all experts, and shows some correspondence between routing and regimes.

Several limitations matter for the next research decision:

1. **Task diversity:** all runs use the same three block-diagonal rotation matrices. There is no nonlinear generator, real dataset, change in dimension, or held-out transition law.
2. **Tuning fairness:** learning rate, epoch count, router size, top-k, and balance coefficient have not been searched with equal validation budgets across models.
3. **Control differences:** the random gate has a smaller, linear architecture, and the oracle uses one expert plus privileged labels. Neither is an exactly capacity-and-information-matched ablation of the learned top-2 router.
4. **Uncertainty:** three seeds provide descriptive variation, not strong statistical evidence. Data and initialization effects are not separately estimated, and test windows are correlated.
5. **Compute:** parameter proxies and incidental training timings do not establish inference speed, FLOP efficiency, memory efficiency, or scalability.
6. **Stability:** true dynamics contract, but neural predictors are unconstrained. Stable data do not guarantee stable recursive neural forecasts.
7. **Interpretability:** top-expert agreement does not identify causal features, disentangled representations, or unique dynamical mechanisms.
8. **Artifact management:** caches and reports require deliberate regeneration after code changes; the project is a research prototype rather than a complete experiment-tracking service.

A useful next stage would first separate data seeds from initialization seeds and give each family a matched validation-tuning budget. A context-length sweep would test how much historical information is needed. Comparing top-1, top-2, and all-three-expert mixtures would separate selection sparsity from mixture modeling. A frozen router with the same architecture as the learned router would make the routing ablation cleaner.

For the dynamics question, a global linear regression baseline and an analytical switching-model/filter reference would clarify how much neural complexity is necessary. An exact-matrix multi-step oracle would separate unavoidable noise accumulation from learned-oracle rollout drift. None of these additions is implemented in the current program.

Only after that foundation is understood should SAE or DAS be added. An SAE could analyze collected hidden activations and their reconstructability; DAS could test whether selected representation subspaces support specified causal interventions. Those would require their own datasets of activations, controls, and validation. The current routing plots do not perform either analysis.

## 22. Glossary and source index

| Term | Meaning in this project |
|---|---|
| Regime | One of the three true transition laws used by the generator |
| Expert | A trainable predictor inside the MoE; not automatically identical to a regime |
| Router/gate | Network or rule selecting expert contributions from the input |
| Sparse computation | Only selected experts process a given input |
| Dense model | One MLP used for every input |
| Context/window | Eight consecutive observed states flattened into the input |
| Teacher-forced one-step evaluation | A fresh actual observed history supplies every prediction |
| Free-running rollout | Predictions become inputs to subsequent predictions |
| Oracle | A model or reference with information unavailable to the ordinary predictor |
| Checkpoint | Saved selected weights and metadata |
| Seed | Number controlling a reproducible random initialization/stream |
| Censoring | An episode ends without an observed qualifying recovery |
| Parameter matching | Choosing model size to approximate a specified parameter count |
| Provenance | Records connecting results to configuration, data, software, and source code |

The central code and evidence are available at:

- [Project README](D:/HarunoKen/Twilight_MoE_Dynamics/README.md)
- [Research plan](D:/HarunoKen/Twilight_MoE_Dynamics/research_plan.md)
- [Full experiment configuration](D:/HarunoKen/Twilight_MoE_Dynamics/configs/experiment.yaml)
- [Data generator](D:/HarunoKen/Twilight_MoE_Dynamics/src/twilight_moe/dynamics.py)
- [Data and window construction](D:/HarunoKen/Twilight_MoE_Dynamics/src/twilight_moe/data.py)
- [Models](D:/HarunoKen/Twilight_MoE_Dynamics/src/twilight_moe/models.py)
- [Routing](D:/HarunoKen/Twilight_MoE_Dynamics/src/twilight_moe/routers.py)
- [Training](D:/HarunoKen/Twilight_MoE_Dynamics/src/twilight_moe/training.py)
- [Evaluation](D:/HarunoKen/Twilight_MoE_Dynamics/src/twilight_moe/evaluation.py)
- [Experiment orchestration](D:/HarunoKen/Twilight_MoE_Dynamics/src/twilight_moe/experiment.py)
- [Comparison notebook](D:/HarunoKen/Twilight_MoE_Dynamics/notebooks/05_compare_performance.ipynb)
- [Full-run metrics](D:/HarunoKen/Twilight_MoE_Dynamics/results/metrics/experiment_57adae732ea0/summary.csv)
- [Source and environment manifest](D:/HarunoKen/Twilight_MoE_Dynamics/results/metrics/experiment_57adae732ea0/environment.json)
- [Experiment log](D:/HarunoKen/Twilight_MoE_Dynamics/results/experiment_log.md)
- [Verification record](D:/HarunoKen/Twilight_MoE_Dynamics/VERIFICATION.md)

No external literature claims are needed to read this report. Mathematical interpretations have been derived from the supplied generator, and numerical findings refer to the saved experiment identified at the top of the document.
