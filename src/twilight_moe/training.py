import copy
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .models import build_model


def seed_everything(seed, threads=2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def fit(name, cfg, seed, train, validation):
    settings = cfg["training"]
    seed_everything(seed, settings["threads"])
    model = build_model(name, cfg)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                 lr=settings["learning_rate"])
    loader = DataLoader(train, batch_size=settings["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(validation, batch_size=4096)
    best, best_state, best_epoch, history = float("inf"), None, 0, []
    start = time.perf_counter()
    for epoch in range(1, settings["epochs"]+1):
        model.train()
        total = 0.
        for x, y, r in loader:
            optimizer.zero_grad()
            prediction, balance, _ = model(x, r if name == "oracle" else None)
            mse = (prediction-y).square().mean()
            use_balance = name in ("moe", "attn_moh", "attn_ffn_moe", "attn_both")
            loss = mse + (settings["balance_weight"]*balance if use_balance else 0)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += mse.item()*len(x)
        model.eval()
        val_total = 0.
        with torch.no_grad():
            for x, y, r in val_loader:
                prediction = model(x, r if name == "oracle" else None)[0]
                val_total += (prediction-y).square().mean().item()*len(x)
        val = val_total / len(validation)
        history.append({"epoch": epoch, "train_mse_normalized": total/len(train),
                        "val_mse_normalized": val})
        if val < best:
            best, best_state, best_epoch = val, copy.deepcopy(model.state_dict()), epoch
    model.load_state_dict(best_state)
    model.eval()
    return model, history, {"best_epoch": best_epoch, "best_val_mse_normalized": best,
                             "training_seconds": time.perf_counter()-start}
