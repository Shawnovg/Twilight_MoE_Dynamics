import torch
from torch import nn

from .routers import Router, sparse_gates

MODEL_NAMES = ("dense_active", "dense_total", "moe", "random", "oracle")


def mlp(inputs, hidden):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 4))


def nparameters(model):
    return sum(p.numel() for p in model.parameters())


class Dense(nn.Module):
    def __init__(self, inputs, hidden):
        super().__init__()
        self.network = mlp(inputs, hidden)

    def forward(self, x, regime=None):
        return self.network(x), x.new_zeros(()), None

    def capacity(self):
        n = nparameters(self)
        return {"total_parameters": n, "trainable_parameters": n, "active_parameters": n}


class MoE(nn.Module):
    def __init__(self, inputs, hidden, router_hidden, experts=3, top_k=2, mode="moe"):
        super().__init__()
        self.mode, self.top_k = mode, (1 if mode == "oracle" else top_k)
        self.experts = nn.ModuleList([mlp(inputs, hidden) for _ in range(experts)])
        self.router = None if mode == "oracle" else Router(inputs, router_hidden, experts, mode == "random")

    def forward(self, x, regime=None):
        if self.mode == "oracle":
            if regime is None:
                raise ValueError("Oracle requires true regimes")
            indices, weights = regime[:, None], x.new_ones((len(x), 1))
            balance = x.new_zeros(())
        else:
            probabilities = self.router(x)
            indices, weights = sparse_gates(probabilities, self.top_k)
            fractions = torch.nn.functional.one_hot(indices, len(self.experts)).float().mean((0, 1))
            balance = len(self.experts) * (fractions.detach() * probabilities.mean(0)).sum()
        prediction = x.new_zeros((len(x), 4))
        for expert_id, expert in enumerate(self.experts):
            rows, slots = torch.where(indices == expert_id)
            if rows.numel():
                prediction = prediction.index_add(0, rows, expert(x[rows]) * weights[rows, slots, None])
        return prediction, balance, indices

    def capacity(self):
        router_count = 0 if self.router is None else nparameters(self.router)
        return {"total_parameters": nparameters(self),
                "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "active_parameters": router_count + self.top_k * nparameters(self.experts[0])}


def build_model(name, cfg):
    from .attention import ATTENTION_NAMES, AttentionPredictor
    if name in ATTENTION_NAMES:
        return AttentionPredictor(cfg, name)
    m = cfg["model"]
    inputs = cfg["data"]["context"] * 4
    if name not in MODEL_NAMES:
        raise ValueError(name)
    if name.startswith("dense"):
        expert = (inputs + 1)*m["hidden"] + (m["hidden"]+1)*m["hidden"] + (m["hidden"]+1)*4
        router = (inputs+1)*m["router_hidden"] + (m["router_hidden"]+1)*m["experts"]
        target = router + (m["top_k"] if name == "dense_active" else m["experts"])*expert
        count = lambda h: (inputs+1)*h + (h+1)*h + (h+1)*4
        width = min(range(1, 4097), key=lambda h: abs(count(h)-target))
        return Dense(inputs, width)
    return MoE(inputs, mode=name, **m)


def configured_models(cfg):
    return tuple(cfg.get("models", MODEL_NAMES))
