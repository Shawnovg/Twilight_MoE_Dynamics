import copy

import numpy as np
import pytest
import torch

from twilight_moe.config import load_config
from twilight_moe.data import load_data, windows, ensure_data
from twilight_moe.dynamics import generate_sequences, transition_matrices
from twilight_moe.evaluation import switch_recovery, routing_statistics, rollout
from twilight_moe.models import MODEL_NAMES, build_model
from twilight_moe.training import fit


def test_stability_and_reproducibility():
    assert np.all(np.linalg.norm(transition_matrices(), ord=2, axis=(1, 2)) < 1)
    args = dict(n=8, length=80, burn_in=32, noise_std=.08, stay_probability=.95, seed=3)
    x, r = generate_sequences(**args)
    xx, rr = generate_sequences(**args)
    np.testing.assert_array_equal(x, xx)
    np.testing.assert_array_equal(r, rr)
    assert x.shape == (8, 81, 4) and r.shape == (8, 80)
    assert np.isfinite(x).all()


def test_transition_residual_matches_generator_noise():
    x, r = generate_sequences(64, 128, 32, .08, .8, 5)
    expected = np.einsum('ntij,ntj->nti', transition_matrices()[r], x[:, :-1])
    residual = x[:, 1:]-expected
    assert abs(residual.mean()) < .002
    assert abs(np.mean(residual**2)-.08**2) < .0002


def test_causal_windows_and_labels():
    x = np.arange(28, dtype=np.float32).reshape(1, 7, 4)
    r = np.array([[0, 1, 2, 0, 1, 2]])
    ds = windows({'states': x, 'regimes': r}, 3, {'mean': np.zeros(4), 'std': np.ones(4)})
    histories, targets, labels = ds.tensors
    np.testing.assert_array_equal(histories[0], x[0, :3].ravel())
    np.testing.assert_array_equal(targets[0], x[0, 3])
    assert labels.tolist() == [2, 0, 1, 2]
    assert len(ds) == 4


def test_data_splits_and_training_only_normalization(tmp_path):
    cfg = load_config()
    cfg['data_root'] = str(tmp_path)
    data, norm = load_data(cfg, 5)
    assert not np.array_equal(data['train']['states'][0], data['val']['states'][0])
    assert not np.array_equal(data['val']['states'][0], data['test']['states'][0])
    np.testing.assert_allclose(norm['mean'], data['train']['states'].mean((0, 1)))
    path = ensure_data(cfg, 5)
    with (path/'test.npz').open('ab') as file:
        file.write(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        ensure_data(cfg, 5)


@pytest.mark.parametrize('k', [1, 2, 3])
def test_router_task_gradients_and_sparse_dispatch(k):
    cfg = load_config()
    cfg['model']['top_k'] = k
    torch.manual_seed(7)
    model = build_model('moe', cfg)
    x = torch.randn(15, cfg['data']['context']*4)
    seen = [0, 0, 0]
    handles = []
    for i, expert in enumerate(model.experts):
        def hook(module, args, output, index=i):
            seen[index] += len(args[0])
        handles.append(expert.register_forward_hook(hook))
    y, _, indices = model(x)
    assert y.shape == (15, 4) and indices.shape == (15, k)
    assert sum(seen) == 15*k
    y.square().mean().backward()
    assert sum(p.grad.abs().sum().item() for p in model.router.parameters()) > 0
    for handle in handles:
        handle.remove()


def test_controls_and_capacity():
    cfg = load_config()
    torch.manual_seed(1)
    random = build_model('random', cfg)
    assert all(not p.requires_grad for p in random.router.parameters())
    x = torch.randn(12, cfg['data']['context']*4)
    assert torch.equal(random(x)[2], random(x)[2])
    oracle = build_model('oracle', cfg)
    with pytest.raises(ValueError):
        oracle(x)
    regimes = torch.arange(12)%3
    assert torch.equal(oracle(x, regimes)[2].flatten(), regimes)
    capacity = {n: build_model(n, cfg).capacity() for n in MODEL_NAMES}
    for name, key in [('dense_active', 'active_parameters'), ('dense_total', 'total_parameters')]:
        assert abs(capacity[name][key]/capacity['moe'][key]-1) < .05


def test_recovery_censoring():
    r = np.array([[0, 0, 1, 1, 1, 1, 2, 2]])
    error = np.array([[9., 9., .1, .1, .1, 9., 9.]])  # context=2 starts at t=1
    result = switch_recovery(error, r, 2, .5, 3)
    assert result['switch_episodes'] == 2
    assert result['recovered_episodes'] == 1
    assert result['recovered_fraction'] == .5 and result['recovery_delay_steps'] == 1


def test_route_permutation_uses_validation():
    val = np.array([[[1], [2], [0]]])
    truth = np.array([[0, 1, 2]])
    stats, counts, utilization = routing_statistics(val, val, truth, truth)
    assert stats['routing_agreement'] == 1
    assert counts.sum() == 3 and np.isclose(utilization.sum(), 1)


def test_fit_deterministic_and_checkpoint_selection(tmp_path):
    cfg = load_config()
    cfg['data_root'] = str(tmp_path)
    cfg['training']['epochs'] = 2
    cfg['data'].update(train_sequences=4, val_sequences=2, test_sequences=2, length=32)
    data, norm = load_data(cfg, 3)
    train, val = [windows(data[s], cfg['data']['context'], norm) for s in ['train', 'val']]
    first, history, metadata = fit('moe', cfg, 3, train, val)
    second, _, _ = fit('moe', cfg, 3, train, val)
    for key in first.state_dict():
        assert torch.equal(first.state_dict()[key], second.state_dict()[key])
    assert metadata['best_val_mse_normalized'] == min(h['val_mse_normalized'] for h in history)


def test_rollout_feeds_back_predictions():
    class Increment(torch.nn.Module):
        def forward(self, x, regime=None):
            return x[:, -4:]+1, torch.tensor(0.), None
    states = np.repeat(np.arange(11, dtype=np.float32)[None, :, None], 4, axis=2)
    split = {'states': states, 'regimes': np.zeros((1, 10), dtype=np.int64)}
    mse, persistence, starts = rollout(Increment(), split, 3,
        {'mean': np.zeros(4), 'std': np.ones(4)}, 4)
    np.testing.assert_allclose(mse, 0)
    np.testing.assert_allclose(persistence, [1, 4, 9, 16])
    assert starts == 2
