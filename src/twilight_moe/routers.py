import torch
from torch import nn


class Router(nn.Module):
    def __init__(self, inputs, hidden, experts, random=False):
        super().__init__()
        self.random = random
        self.network = (nn.Linear(inputs, experts, bias=False) if random else
                        nn.Sequential(nn.Linear(inputs, hidden), nn.Tanh(), nn.Linear(hidden, experts)))
        if random:
            for parameter in self.parameters():
                parameter.requires_grad_(False)

    def forward(self, x):
        return self.network(x).softmax(dim=-1)


def sparse_gates(probabilities, k):
    values, indices = probabilities.topk(k, dim=-1)
    if k == 1:
        # Forward weight 1, surrogate derivative through the selected probability.
        weights = torch.ones_like(values) + values - values.detach()
    else:
        weights = values / values.sum(dim=-1, keepdim=True)
    return indices, weights
