import copy
import json

import numpy as np
import pytest
import torch
from torch import nn

from twilight_moe.attention import ATTENTION_NAMES, HeadAttention
from twilight_moe.config import load_config
from twilight_moe.data import load_data, windows
from twilight_moe.models import build_model
from twilight_moe.training import fit


def test_dense_attention_matches_pytorch_and_is_causal():
    torch.manual_seed(13)
    attention = HeadAttention(16, 4, 4)
    reference = nn.MultiheadAttention(16, 4, bias=False, batch_first=True)
    with torch.no_grad():
        reference.in_proj_weight.copy_(attention.qkv.weight)
        reference.out_proj.weight.copy_(attention.output.weight)
    x = torch.randn(3, 8, 16)
    expected = reference(x, x, x, attn_mask=torch.ones(8, 8, dtype=torch.bool).triu(1))[0]
    actual = attention(x)[0]
    torch.testing.assert_close(actual, expected)
    changed = x.clone()
    changed[:, 4:] += 5
    torch.testing.assert_close(actual[:, :4], attention(changed)[0][:, :4])


def test_moh_gates_and_head_output_formula():
    torch.manual_seed(4)
    model = HeadAttention(16, 4, 4, routing="learned")
    x = torch.randn(3, 8, 16)
    y, balance, detail = model(x, True)
    weights = detail['weights']
    assert torch.all((weights > 0).sum(-1) == 3)
    assert torch.all(weights[..., 0] > 0)  # shared head
    assert torch.all(weights.sum(-1) < 1)  # full-softmax masking, paper Eq.5
    q, k, v = model.qkv(x).reshape(3, 8, 3, 4, 4).permute(2, 0, 3, 1, 4).unbind(0)
    raw = (detail['attention'] @ v).transpose(1, 2)
    outputs = [nn.functional.linear(raw[:, :, h], model.output.weight[:, h*4:(h+1)*4]) for h in range(4)]
    expected = sum(weights[..., h, None]*outputs[h] for h in range(4))
    torch.testing.assert_close(y, expected)
    (y.square().mean()+.01*balance).backward()
    assert model.route_gate.weight.grad.abs().sum() > 0
    assert model.type_gate.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize('family', ATTENTION_NAMES)
def test_attention_families_ignore_labels_and_report_actual_compute(family):
    cfg = load_config('configs/attention_smoke.yaml')
    model = build_model(family, cfg)
    x = torch.randn(5, 32)
    first = model(x)[0]
    torch.testing.assert_close(first, model(x, torch.arange(5)%3)[0])
    assert first.shape == (5, 4)
    assert model.capacity()['attention_dense_compute'] is True
    if family == 'attn_random':
        for block in model.blocks:
            assert not any(p.requires_grad for p in block.attention.route_gate.parameters())
    if family in ('attn_dense', 'attn_moh', 'attn_random'):
        assert model.capacity()['active_parameters'] == model.capacity()['total_parameters']


def test_backbone_initialization_and_total_capacity_controls():
    cfg = load_config('configs/attention_smoke.yaml')
    models = {}
    for name in ATTENTION_NAMES:
        torch.manual_seed(7)
        models[name] = build_model(name, cfg)
    dense = models['attn_dense'].state_dict()
    for name in ['attn_moh', 'attn_random', 'attn_ffn_moe', 'attn_both']:
        other = models[name].state_dict()
        for key in dense:
            torch.testing.assert_close(dense[key], other[key], rtol=0, atol=0)
    for control, treatment in [('attn_total','attn_moh'),('attn_ffn_total','attn_ffn_moe')]:
        assert abs(models[control].capacity()['total_parameters']/models[treatment].capacity()['total_parameters']-1)<.02


def test_attention_fit_checkpoint_and_determinism(tmp_path):
    cfg = load_config('configs/attention_smoke.yaml')
    cfg['data_root'] = str(tmp_path)
    cfg['data'].update(train_sequences=3, val_sequences=2, test_sequences=2, length=32)
    cfg['training']['epochs'] = 2
    data, norm = load_data(cfg, 11)
    train, val = [windows(data[k], 8, norm) for k in ('train','val')]
    first, history, result = fit('attn_moh', cfg, 11, train, val)
    second, _, _ = fit('attn_moh', cfg, 11, train, val)
    assert result['best_val_mse_normalized'] == min(row['val_mse_normalized'] for row in history)
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)


def test_provenance_rejects_one_stale_family_even_with_new_manifest(tmp_path, monkeypatch):
    from twilight_moe import experiment
    current = {'attention.py': 'current-hash'}
    monkeypatch.setattr(experiment, 'run_dir', lambda cfg: tmp_path)
    monkeypatch.setattr(experiment, 'source_hashes', lambda: current)
    cfg = {'seeds': [7], 'models': ['attn_dense', 'attn_moh']}
    (tmp_path/'environment.json').write_text(json.dumps({'source_files': current}))
    (tmp_path/'attn_dense_seed7.json').write_text(json.dumps({'source_files': current}))
    (tmp_path/'attn_moh_seed7.json').write_text(json.dumps({'source_files': {'attention.py': 'old-hash'}}))
    with pytest.raises(ValueError, match='Stale provenance'):
        experiment.verify_provenance(cfg)
    (tmp_path/'attn_moh_seed7.json').write_text(json.dumps({'source_files': current}))
    assert experiment.verify_provenance(cfg)
