import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

from .config import fingerprint
from .dynamics import generate_sequences, transition_matrices

DATA_VERSION = 1


def dataset_path(cfg, seed):
    spec = {"version": DATA_VERSION, "data": cfg["data"], "seed": seed}
    return Path(cfg["data_root"]) / "generated" / fingerprint(spec)


def ensure_data(cfg, seed):
    root = Path(cfg["data_root"])
    for folder in ("raw", "generated", "processed", "metadata"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    path = dataset_path(cfg, seed)
    path.mkdir(exist_ok=True)
    spec = {"version": DATA_VERSION, "data": cfg["data"], "seed": seed}
    metadata_path = root / "metadata" / (path.name + ".json")
    splits = ("train", "val", "test")
    if metadata_path.exists() and all((path / (s + ".npz")).exists() for s in splits):
        metadata = json.loads(metadata_path.read_text())
        for s in splits:
            digest = hashlib.sha256((path / (s + ".npz")).read_bytes()).hexdigest()
            if digest != metadata["sha256"][s]:
                raise ValueError(f"Dataset checksum mismatch: {s}")
        return path
    children = np.random.SeedSequence(seed).spawn(3)
    hashes = {}
    for split, child in zip(splits, children):
        d = cfg["data"]
        states, regimes = generate_sequences(
            d[split + "_sequences"], d["length"], d["burn_in"],
            d["noise_std"], d["stay_probability"], child)
        file = path / (split + ".npz")
        np.savez_compressed(file, states=states, regimes=regimes)
        hashes[split] = hashlib.sha256(file.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps({**spec, "sha256": hashes,
        "matrices": transition_matrices().tolist(),
        "source": "Locally generated synthetic data; independent SeedSequence children",
        "regime_alignment": "r[t] produces x[t+1] from x[t]"}, indent=2))
    return path


def load_data(cfg, seed):
    path = ensure_data(cfg, seed)
    data = {}
    for split in ("train", "val", "test"):
        with np.load(path / (split + ".npz")) as archive:
            data[split] = {k: archive[k] for k in archive.files}
    train = data["train"]["states"]
    normalization = {"mean": train.mean(axis=(0, 1)),
                     "std": train.std(axis=(0, 1)).clip(1e-6)}
    return data, normalization


def windows(split, context, normalization):
    x, r = split["states"], split["regimes"]
    times = range(context - 1, r.shape[1])
    history = np.stack([x[:, t-context+1:t+1] for t in times], axis=1)
    targets = x[:, context:]
    history = (history - normalization["mean"]) / normalization["std"]
    targets = (targets - normalization["mean"]) / normalization["std"]
    return TensorDataset(
        torch.from_numpy(history.reshape(-1, context * 4).astype(np.float32)),
        torch.from_numpy(targets.reshape(-1, 4).astype(np.float32)),
        torch.from_numpy(r[:, context-1:].reshape(-1).copy()))


def regime_ages(regimes):
    ages = np.zeros_like(regimes)
    for t in range(1, regimes.shape[1]):
        ages[:, t] = np.where(regimes[:, t] == regimes[:, t-1], ages[:, t-1] + 1, 0)
    return ages
