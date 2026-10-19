# Twilight MoE Dynamics

A notebook-first PyTorch experiment: can a small sparse Mixture of Experts
predict four-dimensional switching dynamics better than dense baselines?

The attention extension compares MoH-inspired head routing and FFN routing in
the same two-block Transformer, alongside rerun original MLP controls. Start
with notebooks **08–10** for this study. Read
[the attention experiment design](docs/ATTENTION_EXPERIMENT.md) and
[the measured comparison](results/attention_comparison_log.md).

The [updated technical report](docs/TECHNICAL_REPORT.md) covers both experiments.
The [updated PowerPoint](presentations/Twilight_MoE_Dynamics_Research_Updated.pptx)
includes the attention papers, methods and controlled comparisons in 58 slides.

```powershell
$env:TWILIGHT_CONFIG = 'configs/attention_experiment.yaml'
.\.venv\Scripts\python.exe -m jupyter lab notebooks/08_train_attention_comparison.ipynb
```

Notebook 08 checks complete per-model source provenance before reusing results.
Set `FORCE_RETRAIN=True` in its training cell to rerun all families. The default
for notebooks 08–10 is `configs/attention_smoke.yaml`, which only validates the
workflow. The MoH reference computes attention densely and makes no speed claim.

For the complete mathematical and implementation walkthrough, read
[the technical report](D:/HarunoKen/Twilight_MoE_Dynamics/docs/TECHNICAL_REPORT.md).

Code: `D:/HarunoKen/Twilight_MoE_Dynamics`  
Datasets: `E:/HarunoKen/Twilight_MoE_Dynamics`

## Start

The project-local `.venv` is already installed and verified on this computer.
To begin immediately, run `scripts/start_jupyter.ps1` in PowerShell, or:

```powershell
.\.venv\Scripts\python.exe -m jupyter lab notebooks/00_environment_check.ipynb
```

For a fresh installation on another computer, use Python 3.12 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyter lab
```

Open `notebooks/00_environment_check.ipynb`, then work through notebooks 01–07.
Every figure has its own editable Python cell and is saved in `results/figures`.
The notebooks default to the small CPU smoke configuration. They share the
`TWILIGHT_CONFIG` environment variable; set it before launching Jupyter to use
`configs/experiment.yaml` for the larger three-seed experiment.

The saved outputs in notebooks 00–07 describe the original three-seed experiment.
The original PowerPoint and archived report preserve the first study. Current-source paired
comparisons use the attention experiment, which reruns all original families.
To rerun that preset interactively:

```powershell
$env:TWILIGHT_CONFIG = 'configs/experiment.yaml'
.\.venv\Scripts\python.exe -m jupyter lab notebooks/00_environment_check.ipynb
```

For command-line smoke validation:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts/run_experiment.py --config configs/smoke.yaml
.\.venv\Scripts\python.exe scripts/execute_notebooks.py --config configs/smoke.yaml
```

After a full run, refresh the result log with:

```powershell
.\.venv\Scripts\python.exe scripts/summarize_results.py --config configs/experiment.yaml
```

Read `results/experiment_log.md` for observed results and `VERIFICATION.md` for
the execution and inspection record. The initial full preset found higher
one-step error for learned MoE than both dense controls in all three seeds.

The command-line experiment and notebooks use identical implementation code.
`requirements-lock.txt` records the verified environment. Its CPU-only PyTorch
wheel can be installed with `--extra-index-url https://download.pytorch.org/whl/cpu`.
No public data download is needed. Notebook 01 generates all datasets on E:.
`TWILIGHT_DATA_DIR` may override the dataset root. Changing data parameters or
seed produces a separate fingerprinted dataset. Metrics and checkpoints are
stored by configuration fingerprint to avoid mixing experiment settings.

## Reading the experiment

Five models: dense approximately matched for active parameters, dense matched
for total parameters, learned top-k MoE, fixed random-router MoE, and oracle MoE.
Persistence and the known true-dynamics predictor provide additional reference
errors. The oracle receives otherwise unavailable regime information and is an
informational reference, not a guaranteed numerical upper bound after training.

All models observe causal windows. Train/validation/test are independently
generated sequences, and normalization uses training states only. Validation
selects checkpoints; test data are evaluated afterward. Metrics use original
state units. See `research_plan.md` for assumptions, adaptation definitions,
compute caveats, and the later research roadmap.
