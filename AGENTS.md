# Twilight MoE Dynamics

Code and results live in this project. All generated datasets belong under
`E:/HarunoKen/Twilight_MoE_Dynamics` (override with `TWILIGHT_DATA_DIR`).
Keep the workflow runnable as numbered Jupyter notebooks with short cells and
visible plotting code. Put reusable algorithms in `src/twilight_moe`.

This is a performance-first controlled experiment, not a novelty claim.
Never expose regime labels to non-oracle models, fit preprocessing on validation
or test data, or split overlapping windows across dataset partitions.
Select checkpoints on validation MSE only. Preserve negative results.
Distinguish smoke runs from substantive evidence and parameter proxies from FLOPs.
Run `python -m pytest` and the notebook smoke workflow after algorithm changes.
Do not implement SAE/DAS interventions in version 1.
Increment DATA_VERSION when generator behavior changes. Before comparing cached
results after source edits, rerun all model families and check source hashes in
the saved environment manifest. Regenerate the log with summarize_results.py.
