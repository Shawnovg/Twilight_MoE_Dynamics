"""Descriptive head diagnostics; labels are used only after inference."""
import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import windows
from .evaluation import routing_statistics


@torch.no_grad()
def collect_attention(model, split, context, normalization):
    model.eval()
    route_batches, ffn_batches, weight_sums, attention_sums = [], [], None, None
    token_count, window_count = 0, 0
    for histories, _, _ in DataLoader(windows(split, context, normalization), batch_size=512):
        _, _, details = model.forward_features(histories, return_details=True)
        weights = torch.stack([item["weights"] for item in details])  # L B T H
        attention = torch.stack([item["attention"][:, :, -1] for item in details])  # L B H T
        ws, ats = weights.sum((1, 2)).numpy(), attention.sum(1).numpy()
        weight_sums = ws if weight_sums is None else weight_sums + ws
        attention_sums = ats if attention_sums is None else attention_sums + ats
        if details[0]["routes"] is not None:
            route_batches.append(np.stack([item["routes"][:, -1].numpy() for item in details], axis=1))
        if details[0]["ffn_routes"] is not None:
            ffn_batches.append(np.stack([item["ffn_routes"][:, -1].numpy() for item in details], axis=1))
        token_count += len(histories)*context
        window_count += len(histories)
    return {"head_weight_mean": weight_sums/token_count,
            "last_query_attention_mean": attention_sums/window_count,
            "head_routes": np.concatenate(route_batches) if route_batches else None,
            "ffn_routes": np.concatenate(ffn_batches) if ffn_batches else None}


def evaluate_attention(model, cfg, data, normalization):
    c = cfg["data"]["context"]
    test = collect_attention(model, data["test"], c, normalization)
    val = collect_attention(model, data["val"], c, normalization)
    metrics, arrays = {}, {k: v for k, v in test.items() if v is not None}
    for kind in ("head", "ffn"):
        key = kind+"_routes"
        if test[key] is None:
            continue
        for layer in range(test[key].shape[1]):
            tr, vr = test[key][:, layer], val[key][:, layer]
            # The three routed heads/FFN experts are arbitrary IDs. Shared heads
            # are not classified as regimes. Mapping is fit on validation only.
            stats, confusion, utilization = routing_statistics(vr, tr,
                data["val"]["regimes"][:, c-1:], data["test"]["regimes"][:, c-1:])
            prefix = f"{kind}_layer{layer}"
            metrics[prefix+"_agreement"] = stats["routing_agreement"]
            metrics[prefix+"_mapping_validation"] = stats["route_mapping_validation"]
            arrays[prefix+"_confusion"] = confusion
            arrays[prefix+"_utilization"] = utilization
    return metrics, arrays
