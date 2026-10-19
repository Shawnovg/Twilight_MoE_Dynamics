from itertools import permutations

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import regime_ages, windows
from .dynamics import transition_matrices


@torch.no_grad()
def predict(model, split, context, normalization):
    predictions, routes = [], []
    model.eval()
    for x, _, r in DataLoader(windows(split, context, normalization), batch_size=4096):
        p, _, selected = model(x, r if getattr(model, "mode", None) == "oracle" else None)
        predictions.append(p.numpy()*normalization["std"] + normalization["mean"])
        if selected is not None:
            routes.append(selected.numpy())
    shape = split["regimes"][:, context-1:].shape
    predictions = np.concatenate(predictions).reshape(*shape, 4)
    error = ((predictions-split["states"][:, context:])**2).mean(axis=-1)
    selected = np.concatenate(routes).reshape(*shape, -1) if routes else None
    return error, selected


def switch_recovery(errors, regimes, context, threshold, consecutive=3):
    """Only actual observed switch episodes; delay is retrospective run start."""
    delays, episode_count = [], 0
    for sequence, r in enumerate(regimes):
        switches = np.flatnonzero(r[1:] != r[:-1])+1
        for j, switch in enumerate(switches):
            if switch < context-1:
                continue
            end = switches[j+1] if j+1 < len(switches) else len(r)
            values = errors[sequence, switch-context+1:end-context+1]
            episode_count += 1
            for delay in range(len(values)-consecutive+1):
                if np.all(values[delay:delay+consecutive] <= threshold):
                    delays.append(delay)
                    break
    return {"switch_episodes": episode_count, "recovered_episodes": len(delays),
            "recovered_fraction": len(delays)/episode_count if episode_count else float("nan"),
            "recovery_delay_steps": float(np.mean(delays)) if delays else float("nan"),
            "recovery_threshold": float(threshold)}


def event_aligned(errors, regimes, context, window):
    bins = [[] for _ in range(window)]
    for sequence, r in enumerate(regimes):
        switches = np.flatnonzero(r[1:] != r[:-1])+1
        for j, switch in enumerate(switches):
            if switch < context-1:
                continue
            end = switches[j+1] if j+1 < len(switches) else len(r)
            for lag in range(min(window, end-switch)):
                bins[lag].append(errors[sequence, switch-context+1+lag])
    return np.array([np.mean(b) if b else np.nan for b in bins]), np.array([len(b) for b in bins])


def routing_statistics(val_routes, test_routes, val_regimes, test_regimes, experts=3):
    val_top = val_routes[..., 0].ravel()
    val_truth = val_regimes.ravel()
    mapping = max(permutations(range(experts)),
                  key=lambda p: np.mean(np.asarray(p)[val_top] == val_truth))
    top = test_routes[..., 0].ravel()
    truth = test_regimes.ravel()
    confusion = np.zeros((experts, experts), dtype=int)
    np.add.at(confusion, (truth, top), 1)
    utilization = np.bincount(test_routes.ravel(), minlength=experts)/test_routes.size
    return {"routing_agreement": float(np.mean(np.asarray(mapping)[top] == truth)),
            "route_mapping_validation": list(mapping)}, confusion, utilization


@torch.no_grad()
def rollout(model, split, context, normalization, horizon):
    # Nonoverlapping rollout targets within each sequence reduce redundant work.
    states, regimes = split["states"], split["regimes"]
    origins = np.arange(context-1, regimes.shape[1]-horizon+1, horizon)
    histories = np.stack([states[:, t-context+1:t+1] for t in origins], axis=1)
    hist = torch.from_numpy(((histories-normalization["mean"])/normalization["std"])
                            .reshape(-1, context, 4).astype(np.float32))
    errors, persistence = [], []
    for h in range(1, horizon+1):
        r = torch.from_numpy(regimes[:, origins+h-1].reshape(-1).copy())
        prediction = model(hist.flatten(1), r if getattr(model, "mode", None) == "oracle" else None)[0]
        actual = states[:, origins+h].reshape(-1, 4)
        original = prediction.numpy()*normalization["std"]+normalization["mean"]
        errors.append(float(((original-actual)**2).mean()))
        persistence.append(float(((histories[:, :, -1].reshape(-1, 4)-actual)**2).mean()))
        hist = torch.cat([hist[:, 1:], prediction[:, None]], dim=1)
    return np.asarray(errors), np.asarray(persistence), len(hist)


def evaluate(model, cfg, data, normalization):
    c, e = cfg["data"]["context"], cfg["evaluation"]
    val_error, val_routes = predict(model, data["val"], c, normalization)
    test_error, test_routes = predict(model, data["test"], c, normalization)
    late = regime_ages(data["val"]["regimes"])[:, c-1:] >= c
    if not late.any():
        raise ValueError("No late-regime validation examples for recovery threshold")
    threshold = e["recovery_multiplier"]*val_error[late].mean()
    ages = regime_ages(data["test"]["regimes"])[:, c-1:]
    test = data["test"]
    current = test["states"][:, c-1:-1]
    target = test["states"][:, c:]
    truth = test["regimes"][:, c-1:]
    known = np.einsum("ntij,ntj->nti", transition_matrices()[truth], current)
    metrics = {"test_mse": float(test_error.mean()),
        "test_windows": test_error.size,
        "switch_mse": float(test_error[ages == 0].mean()) if (ages == 0).any() else float("nan"),
        "persistence_mse": float(((current-target)**2).mean()),
        "known_dynamics_mse": float(((known-target)**2).mean()),
        "noise_variance": cfg["data"]["noise_std"]**2}
    metrics.update(switch_recovery(test_error, test["regimes"], c, threshold, e["recovery_consecutive"]))
    curve, counts = event_aligned(test_error, test["regimes"], c, e["adaptation_window"])
    multi, persistence, starts = rollout(model, test, c, normalization, e["rollout_horizon"])
    metrics.update({"rollout_mean_mse": float(multi.mean()), "rollout_origins": starts})
    arrays = {"test_error": test_error, "adaptation_mse": curve, "adaptation_counts": counts,
              "rollout_mse": multi, "rollout_persistence": persistence}
    if test_routes is not None:
        stats, confusion, utilization = routing_statistics(val_routes, test_routes,
            data["val"]["regimes"][:, c-1:], truth)
        metrics.update(stats)
        arrays.update({"confusion": confusion, "utilization": utilization,
                       "selected_experts": test_routes})
    return metrics, arrays
