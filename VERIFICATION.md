# Verification — 2026-10-08

## Attention extension — 2026-10-09

- `python -m pytest -q`: **25 passed**, including reference-attention equivalence,
  causality, exact gate/output equations, non-oracle label independence,
  router gradients, initialization pairing and stale-family provenance rejection.
- All **11 notebooks** executed top-to-bottom with `configs/attention_smoke.yaml`.
  Separate saved outputs: `results/notebook_smoke_attention`. No notebook error
  outputs, and every code cell has an execution count.
- All **13 smoke checkpoints** reproduce saved one-step MSE within absolute
  tolerance 1e-9. Stored normalizers match training states and checkpoint epochs
  match the minimum validation loss in their histories.
- All eight new smoke figures were inspected directly as saved PNGs.
- Full experiment completed: **39 runs**, 13 families, three seeds and 40 epochs.
  All 39 checkpoints independently reproduced saved test MSE within 1e-9;
  normalization, validation-selected epochs and per-run source hashes matched.
- Notebooks 08–10 executed with the full configuration: all 17 code cells
  completed without error outputs; notebook format validation passed.
  All eight full-run figures were inspected directly as saved PNGs.
- `summarize_results.py` regenerated `results/attention_comparison_log.md`.
  MoH reduced mean paired one-step error by 2.07% versus standard attention,
  but only 0.25% versus its total-parameter control; the frozen router had
  slightly lower mean error than learned MoH. Negative controls are preserved.
- Browser HTML layout review was unavailable: automatic approval review denied
  the localhost browser preview because the approval service reported a usage
  limit. This is not a scientific computation failure. No browser review is claimed.
- The original deck, archived 8 October report and original result directory
  remain historical. The consolidated technical report and updated PowerPoint
  now cover both studies. Current-source comparisons rerun the old families
  alongside the new models.

## Original experiment record

- Python 3.12.14; PyTorch 2.14.1+cpu; isolated project `.venv`.
- `python -m pytest -q`: **12 passed**.
- All eight notebooks executed top-to-bottom using the smoke preset.
- All eight notebooks then executed with `configs/experiment.yaml`: three seeds,
  five models per seed, 40 epochs per model.
- Saved notebooks contain full-preset outputs: 34 executed code cells, nine
  inline figures, and no error outputs. Notebook format validation passed.
- All 15 full-run checkpoints were reloaded independently; recomputed one-step
  test MSE matched saved metrics (absolute tolerance 1e-9).
- All nine full-run figures were visually inspected, with detailed inspection
  of the one-step and rollout comparisons. The rendered notebook HTML summary
  and notebook layout were also inspected in the browser.
- Datasets were generated on E: and verified against recorded SHA-256 hashes.
- The restricted execution sandbox blocks Jupyter's local socket creation.
  Notebook execution succeeded with authorized local execution outside it.
- Learned MoE had higher one-step MSE than both dense controls in all three
  full-run seeds. See `results/experiment_log.md` for paired differences.

The datasets are synthetic. No external research dataset was downloaded.
Active parameter counts are proxies, not FLOP or speed measurements.
Future SAE/DAS interventions are documented but not implemented.
