import hashlib
import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:12]


def load_config(path=None):
    path = Path(path or os.environ.get("TWILIGHT_CONFIG", "configs/smoke.yaml"))
    if not path.is_absolute():
        path = ROOT / path
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["data_root"] = os.environ.get("TWILIGHT_DATA_DIR", cfg["data_root"])
    d, m, t, e = (cfg[k] for k in ("data", "model", "training", "evaluation"))
    assert m["experts"] == 3, "Version 1 has exactly three regimes/experts"
    assert 1 <= m["top_k"] <= m["experts"]
    assert 2 <= d["context"] < d["length"]
    assert 0 < d["stay_probability"] < 1 and d["noise_std"] > 0
    assert all(d[k] > 0 for k in ("train_sequences", "val_sequences", "test_sequences"))
    assert t["epochs"] > 0 and t["batch_size"] > 0
    assert 0 < e["rollout_horizon"] <= d["length"] - d["context"] + 1
    if "attention" in cfg:
        a = cfg["attention"]
        assert a["width"] > 0 and a["head_dim"] > 0 and a["layers"] > 0
        assert 0 < a["shared_heads"] < a["heads"]
        assert a["heads"]-a["shared_heads"] == 3, "Regime-alignment diagnostic assumes 3 routed heads"
        assert 1 <= a["top_k"] <= a["heads"]-a["shared_heads"]
        assert 1 <= a["ffn_top_k"] <= a["ffn_experts"]
    return cfg


def run_dir(cfg):
    path = ROOT / "results" / "metrics" / (cfg["name"] + "_" + fingerprint(cfg))
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_dir(cfg):
    path = ROOT / "results" / "checkpoints" / run_dir(cfg).name
    path.mkdir(parents=True, exist_ok=True)
    return path
