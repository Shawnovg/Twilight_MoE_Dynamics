"""Small causal Transformer controls and a MoH-inspired attention layer.

MoH routing follows Jin et al. (ICML 2025), equations 4--8:
https://proceedings.mlr.press/v267/jin25l.html
This quality-oriented reference computes all Q/K/V and head attention densely,
then gates outputs. Logical head selection is NOT a compute-saving claim.
"""
import hashlib
import math

import torch
from torch import nn

from .routers import sparse_gates

ATTENTION_NAMES = ("attn_dense", "attn_active", "attn_total", "attn_moh",
                   "attn_random", "attn_ffn_moe", "attn_ffn_total", "attn_both")


class HeadAttention(nn.Module):
    def __init__(self, width, heads, head_dim, shared=1, top_k=2, routing="dense"):
        super().__init__()
        self.width, self.heads, self.head_dim = width, heads, head_dim
        self.shared, self.top_k, self.routing = shared, top_k, routing
        self.qkv = nn.Linear(width, 3 * heads * head_dim, bias=False)
        self.output = nn.Linear(heads * head_dim, width, bias=False)
        if routing != "dense":
            self.route_gate = nn.Linear(width, heads-shared, bias=False)
            self.type_gate = nn.Linear(width, 2, bias=False)
            self.shared_gate = nn.Linear(width, shared, bias=False)
            if routing == "random":
                for gate in (self.route_gate, self.type_gate, self.shared_gate):
                    gate.requires_grad_(False)

    def gates(self, x):
        """Return all-head weights, auxiliary loss and routed IDs (excluding shared)."""
        if self.routing == "dense":
            return x.new_ones((*x.shape[:2], self.heads)), x.new_zeros(()), None
        probabilities = self.route_gate(x).softmax(-1)
        values, indices = probabilities.topk(self.top_k, dim=-1)
        selected = torch.zeros_like(probabilities).scatter(-1, indices, values)
        # Paper Eq. 5 retains full-softmax probabilities after top-k masking.
        # In particular, do not silently normalize the selected weights again.
        types = self.type_gate(x).softmax(-1)
        shared = self.shared_gate(x).softmax(-1) * types[..., :1]
        weights = torch.cat((shared, selected * types[..., 1:]), dim=-1)
        mask = torch.zeros_like(probabilities).scatter(-1, indices, 1.)
        frequency = mask.mean((0, 1)).detach()  # fraction of TOKENS, not slots
        balance = (frequency * probabilities.mean((0, 1))).sum()  # Eq. 7
        if self.routing == "random":
            balance = x.new_zeros(())
        return weights, balance, indices

    def forward(self, x, return_details=False):
        batch, length, _ = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        future = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
        attention = scores.masked_fill(future, float("-inf")).softmax(-1)
        values = (attention @ v).transpose(1, 2)
        weights, balance, indices = self.gates(x)
        output = self.output((values * weights[..., None]).reshape(batch, length, -1))
        details = {"weights": weights, "attention": attention, "routes": indices} if return_details else None
        return output, balance, details


class TokenFFN(nn.Module):
    def __init__(self, width, hidden, experts=1, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(width, hidden), nn.GELU(),
                                                  nn.Linear(hidden, width)) for _ in range(experts)])
        self.router = nn.Linear(width, experts, bias=False) if experts > 1 else None

    def forward(self, x):
        if self.router is None:
            return self.experts[0](x), x.new_zeros(()), None
        flat = x.flatten(0, 1)
        probabilities = self.router(flat).softmax(-1)
        indices, weights = sparse_gates(probabilities, self.top_k)
        prediction = torch.zeros_like(flat)
        for i, expert in enumerate(self.experts):
            rows, slots = torch.where(indices == i)
            if rows.numel():
                prediction = prediction.index_add(0, rows, expert(flat[rows]) * weights[rows, slots, None])
        frequencies = nn.functional.one_hot(indices, len(self.experts)).float().mean((0, 1)).detach()
        balance = len(self.experts) * (frequencies * probabilities.mean(0)).sum()
        return prediction.reshape_as(x), balance, indices.reshape(*x.shape[:2], self.top_k)


class TransformerBlock(nn.Module):
    def __init__(self, width, heads, head_dim, shared, top_k, routing, ffn_hidden, ffn_experts, ffn_top_k):
        super().__init__()
        self.norm_attention = nn.LayerNorm(width)
        self.attention = HeadAttention(width, heads, head_dim, shared, top_k, routing)
        self.norm_ffn = nn.LayerNorm(width)
        self.ffn = TokenFFN(width, ffn_hidden, ffn_experts, ffn_top_k)

    def forward(self, x, return_details=False):
        attended, head_balance, details = self.attention(self.norm_attention(x), return_details)
        x = x + attended
        transformed, ffn_balance, ffn_routes = self.ffn(self.norm_ffn(x))
        if return_details:
            details["ffn_routes"] = ffn_routes
        return x + transformed, head_balance + ffn_balance, details


class AttentionPredictor(nn.Module):
    def __init__(self, cfg, family):
        super().__init__()
        a = cfg["attention"]
        self.context, self.family = cfg["data"]["context"], family
        width, heads, head_dim = a["width"], a["heads"], a["head_dim"]
        routing = "random" if family == "attn_random" else (
            "learned" if family in ("attn_moh", "attn_both") else "dense")
        ffn_experts = a["ffn_experts"] if family in ("attn_ffn_moe", "attn_both") else 1
        hidden = a["ffn_hidden"]
        # Match dense-control TOTAL parameters by changing only FFN width.
        if family == "attn_total":
            extra = width * (a["heads"] + 2)
            hidden = min(range(1, 4097), key=lambda h: abs((2*width+1)*(h-hidden)-extra))
        if family == "attn_ffn_total":
            original_ffn = (2*width+1)*hidden + width
            target = a["ffn_experts"]*original_ffn + width*a["ffn_experts"]
            hidden = min(range(1, 4097), key=lambda h: abs((2*width+1)*h+width-target))
        if family == "attn_active":
            heads = a["shared_heads"] + a["top_k"]
        self.embedding = nn.Linear(4, width)
        self.position = nn.Parameter(torch.empty(1, self.context, width))
        self.blocks = nn.ModuleList([TransformerBlock(width, heads, head_dim,
            a["shared_heads"], a["top_k"], routing, hidden, ffn_experts,
            a["ffn_top_k"]) for _ in range(a["layers"])])
        self.final_norm = nn.LayerNorm(width)
        self.readout = nn.Linear(width, 4)
        self.reset_seeded(torch.initial_seed())

    def reset_seeded(self, seed):
        """Name-seeded initialization aligns identical components across ablations."""
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                derived = int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "little")
                generator = torch.Generator().manual_seed(derived % (2**63-1))
                if name == "position":
                    parameter.normal_(0, .02, generator=generator)
                elif name.endswith("bias"):
                    parameter.zero_()
                elif "norm" in name:
                    parameter.fill_(1.)
                else:
                    nn.init.xavier_uniform_(parameter, generator=generator)

    def forward_features(self, x, return_details=False):
        states = x.reshape(-1, self.context, 4)
        hidden = self.embedding(states) + self.position
        balance, details = x.new_zeros(()), []
        for block in self.blocks:
            hidden, auxiliary, detail = block(hidden, return_details)
            balance = balance + auxiliary / len(self.blocks)
            if return_details:
                details.append(detail)
        return self.final_norm(hidden), balance, details

    def forward(self, x, regime=None):
        # regime is deliberately ignored for every attention family.
        hidden, balance, _ = self.forward_features(x)
        return self.readout(hidden[:, -1]), balance, None

    def capacity(self):
        total = sum(p.numel() for p in self.parameters())
        # Attention is dense in this reference; only FFN dispatch skips experts.
        active = total
        for block in self.blocks:
            ffn = block.ffn
            if ffn.router is not None:
                active -= (len(ffn.experts)-ffn.top_k)*sum(p.numel() for p in ffn.experts[0].parameters())
        attention = self.blocks[0].attention
        selected = attention.heads if attention.routing == "dense" else attention.shared+attention.top_k
        return {"total_parameters": total,
                "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "active_parameters": active, "attention_heads": attention.heads,
                "logical_selected_heads": selected, "attention_dense_compute": True}
