import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import ROOT, checkpoint_dir, run_dir
from .data import dataset_path, load_data, windows
from .evaluation import evaluate
from .models import MODEL_NAMES, build_model, configured_models
from .training import fit


def run_models(cfg, names=None):
    names = configured_models(cfg) if names is None else names
    directory = run_dir(cfg)
    (directory / "config.json").write_text(json.dumps(cfg, indent=2))
    (directory / "environment.json").write_text(json.dumps({"python": sys.version,
        "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__,
        "device": "cpu", "source_files": source_hashes()}, indent=2))
    rows = []
    for seed in cfg["seeds"]:
        data, normalization = load_data(cfg, seed)
        context = cfg["data"]["context"]
        train, val = (windows(data[s], context, normalization) for s in ("train", "val"))
        for name in names:
            model, history, training = fit(name, cfg, seed, train, val)
            metrics, arrays = evaluate(model, cfg, data, normalization)
            if name.startswith("attn_"):
                from .attention_diagnostics import evaluate_attention
                extra_metrics, extra_arrays = evaluate_attention(model, cfg, data, normalization)
                metrics.update(extra_metrics)
                arrays.update(extra_arrays)
            stem = f"{name}_seed{seed}"
            row = {"model": name, "seed": seed, **model.capacity(), **training, **metrics,
                   "dataset": str(dataset_path(cfg, seed)), "source_files": source_hashes()}
            (directory / (stem + ".json")).write_text(json.dumps(row, indent=2, allow_nan=True))
            pd.DataFrame(history).to_csv(directory / (stem + "_history.csv"), index=False)
            np.savez_compressed(directory / (stem + "_diagnostics.npz"), **arrays)
            torch.save({"state_dict": model.state_dict(), "model": name, "seed": seed,
                "config": cfg, "source_files": source_hashes(),
                "normalization": {k: v.tolist() for k, v in normalization.items()}},
                checkpoint_dir(cfg) / (stem + ".pt"))
            rows.append(row)
            print(f"{name:16s} seed={seed} test MSE={row['test_mse']:.6f}; best epoch={row['best_epoch']}", flush=True)
    all_rows = load_metrics(cfg)
    all_rows.to_csv(directory / "summary.csv", index=False)
    return pd.DataFrame(rows)


def source_hashes():
    import hashlib
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "src" / "twilight_moe").glob("*.py"))}


def load_metrics(cfg):
    records = []
    for seed in cfg["seeds"]:
        for name in configured_models(cfg):
            file = run_dir(cfg) / f"{name}_seed{seed}.json"
            if file.exists():
                records.append(json.loads(file.read_text()))
    if not records:
        raise FileNotFoundError("Train models in notebooks 03 and 04 first.")
    return pd.DataFrame(records)


def verify_provenance(cfg, require_complete=True):
    """Refuse mixed code revisions or missing families in a reported comparison."""
    current = source_hashes()
    manifest = json.loads((run_dir(cfg)/"environment.json").read_text())
    if manifest["source_files"] != current:
        raise ValueError("Source manifest differs; rerun every family before comparison")
    for seed in cfg["seeds"]:
        for name in configured_models(cfg):
            file = run_dir(cfg)/f"{name}_seed{seed}.json"
            if not file.exists() and not require_complete:
                continue
            row = json.loads(file.read_text())
            if row.get("source_files") != current:
                raise ValueError(f"Stale provenance for {name}, seed {seed}")
    return True


def load_diagnostics(cfg, name, seed):
    with np.load(run_dir(cfg) / f"{name}_seed{seed}_diagnostics.npz") as archive:
        return {k: archive[k] for k in archive.files}


def load_checkpoint(cfg, name, seed):
    saved = torch.load(checkpoint_dir(cfg) / f"{name}_seed{seed}.pt", weights_only=True)
    model = build_model(name, saved["config"])
    model.load_state_dict(saved["state_dict"])
    model.eval()
    normalization = {k: np.array(v, dtype=np.float32) for k, v in saved["normalization"].items()}
    return model, normalization
