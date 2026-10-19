"""Regenerate notebook source. Execution is a separate, explicit step."""
from pathlib import Path
from textwrap import dedent

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SETUP = '''
from pathlib import Path
import os, sys
ROOT = Path.cwd().resolve()
if not (ROOT / 'src' / 'twilight_moe').exists():
    ROOT = ROOT.parent
assert (ROOT / 'src' / 'twilight_moe').exists(), 'Start Jupyter from the project folder'
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from twilight_moe.config import load_config, run_dir
from twilight_moe.plotting import style, save_figure, COLORS
cfg = load_config()
style()
print('Preset:', cfg['name'], '| seeds:', cfg['seeds'])
print('Dataset root:', cfg['data_root'])
'''


def notebook(name, title, intro, sections):
    cells = [nbf.v4.new_markdown_cell(f'# {title}\n\n{intro}\n\n'
             '**Run cells in order.** The default smoke preset validates the pipeline; '
             'it is too small for a scientific conclusion.'),
             nbf.v4.new_markdown_cell('## Setup'), nbf.v4.new_code_cell(dedent(SETUP).strip())]
    for heading, prose, code in sections:
        cells.append(nbf.v4.new_markdown_cell('## ' + heading + '\n\n' + prose))
        if code:
            cells.append(nbf.v4.new_code_cell(dedent(code).strip()))
    nb = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3 (ipykernel)', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.12'}})
    nbf.validate(nb)
    folder = ROOT / 'notebooks'
    folder.mkdir(exist_ok=True)
    nbf.write(nb, folder / (name + '.ipynb'))


notebook('00_environment_check', '00 · Environment and experiment map',
    'Check the interpreter, package versions, configuration, and data location before generating data.', [
    ('Check dependencies', 'The CPU environment is sufficient for both presets.', '''
    import platform, torch, yaml, nbformat
    print('Python:', sys.executable)
    print('Version:', platform.python_version(), '| PyTorch:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available(), '| experiment device: CPU')
    display(pd.Series(cfg['data'], name='Data settings').to_frame())
    display(pd.Series(cfg['training'], name='Training settings').to_frame())
    '''),
    ('Check storage', 'All synthetic datasets are written under the configured E: root.', '''
    data_root = Path(cfg['data_root'])
    data_root.mkdir(parents=True, exist_ok=True)
    print('Code:', ROOT)
    print('Data:', data_root.resolve())
    print('Run results:', run_dir(cfg))
    '''),
    ('Notebook order', '01 generate → 02 inspect → 03 dense training → 04 MoE training '
     '→ 05 compare → 06 routing and adaptation → 07 rollouts.\n\n'
     'Use `TWILIGHT_CONFIG=configs/experiment.yaml` before launching Jupyter for the three-seed preset. '
     'After changing a configuration, rerun the whole sequence. The detailed assumptions are in `research_plan.md`.', None)])

notebook('01_generate_switching_data', '01 · Generate switching dynamics',
    'Generate x[t+1] = A[r[t]] x[t] + noise, with four state dimensions and three Markov regimes.', [
    ('Inspect the generator', 'Regime r[t] acts on x[t]. The norm bound ensures stability even during switching.', '''
    from twilight_moe.dynamics import transition_matrices
    A = transition_matrices()
    print('Matrix operator norms:', np.linalg.norm(A, ord=2, axis=(1, 2)))
    p = cfg['data']['stay_probability']
    P = np.full((3, 3), (1-p)/2)
    np.fill_diagonal(P, p)
    display(pd.DataFrame(P, index=['r0', 'r1', 'r2'], columns=['r0', 'r1', 'r2']))
    '''),
    ('Generate independent sequence splits', 'Independent random streams prevent window leakage. Existing datasets are checksum-verified.', '''
    from twilight_moe.data import ensure_data, load_data
    paths = {seed: str(ensure_data(cfg, seed)) for seed in cfg['seeds']}
    display(pd.Series(paths, name='Dataset path').to_frame())
    data, normalization = load_data(cfg, cfg['seeds'][0])
    display(pd.DataFrame({split: {'sequences': len(d['states']),
                                'transitions_per_sequence': d['regimes'].shape[1]}
                         for split, d in data.items()}).T)
    '''),
    ('Check stability and normalization', 'Normalization statistics are computed from training states only.', '''
    assert all(np.isfinite(d['states']).all() for d in data.values())
    assert np.allclose(normalization['mean'], data['train']['states'].mean(axis=(0, 1)))
    display(pd.DataFrame(normalization, index=[f'x{i}' for i in range(4)]))
    '''),
    ('Provenance', 'The E: metadata folder records generator parameters, seed, matrices, regime alignment, '
     'and SHA-256 hashes. No external dataset is used in version 1.', None)])

notebook('02_inspect_and_visualize_data', '02 · Inspect trajectories and regime coverage',
    'Inspect training data only before model fitting. Each plot can be regenerated independently after setup and loading.', [
    ('Load the training split', 'No test data are used to choose settings.', '''
    from twilight_moe.data import load_data
    data, normalization = load_data(cfg, cfg['seeds'][0])
    x, r = data['train']['states'], data['train']['regimes']
    '''),
    ('Plot one sequence', 'The regime panel identifies the transition applied at time t.', '''
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, height_ratios=[3, 1])
    for j, (color, line) in enumerate(zip(['#315b83', '#b47719', '#965f86', '#606c38'], ['-', '--', ':', '-.'])):
        axes[0].plot(x[0, :, j], color=color, linestyle=line, label=f'x{j}')
    axes[0].set(title='Training sequence 0: four state coordinates', ylabel='State (original units)')
    axes[0].legend(ncol=4)
    axes[1].step(np.arange(len(r[0])), r[0], where='post', color='#315b83')
    axes[1].set(xlabel='Transition time t', ylabel='Regime r[t]', yticks=[0, 1, 2])
    save_figure(fig, cfg, 'training_trajectory')
    plt.show()
    '''),
    ('Plot regime coverage', 'Coverage is measured over training transitions, including all sequences.', '''
    counts = np.bincount(r.ravel(), minlength=3)
    fig, ax = plt.subplots()
    ax.bar(['Regime 0', 'Regime 1', 'Regime 2'], counts, color='#315b83')
    ax.set(title=f'Training regime coverage ({r.size:,} transitions)', ylabel='Transition count')
    save_figure(fig, cfg, 'regime_coverage')
    plt.show()
    print('Observed switch fraction:', np.mean(r[:, 1:] != r[:, :-1]))
    print('Configured switch probability:', 1-cfg['data']['stay_probability'])
    '''),
    ('Interpretation', 'A single trajectory is illustrative. Use aggregate coverage to check for absent regimes. '
     'Different runs need not have exactly equal counts.', None)])

for file, title, names, intro in [
    ('03_train_dense_baselines', '03 · Train dense capacity controls', ['dense_active', 'dense_total'],
     'Train two three-layer dense MLPs, approximately matching the learned MoE active and total parameter counts.'),
    ('04_train_small_moe', '04 · Train sparse MoE and routing controls', ['moe', 'random', 'oracle'],
     'Train the learned top-k router, a fixed random-router control, and a regime-informed oracle.')]:
    notebook(file, title, intro, [
        ('Inspect models', 'Hidden widths and capacity counts are explicit. Active parameters are a proxy, not a FLOP measurement.', f'''
        from twilight_moe.models import build_model
        names = {names!r}
        display(pd.DataFrame([{{'model': name, **build_model(name, cfg).capacity()}} for name in names]))
        print(build_model(names[0], cfg))
        '''),
        ('Train and select checkpoints', 'The selected epoch minimizes validation MSE. Test evaluation occurs after selection. '
         'Losses for training are normalized; reported test MSE is in original state units.', '''
        from twilight_moe.experiment import run_models
        rows = run_models(cfg, names)
        display(rows[['model', 'seed', 'best_epoch', 'test_mse', 'training_seconds']].round(6))
        '''),
        ('Plot validation learning curves', 'The first seed is shown. Each model has a separate history CSV for every seed.', '''
        seed = cfg['seeds'][0]
        fig, ax = plt.subplots()
        for name in names:
            history = pd.read_csv(run_dir(cfg) / f'{name}_seed{seed}_history.csv')
            ax.plot(history['epoch'], history['val_mse_normalized'], marker='o',
                    label=name, color=COLORS[name])
        ax.set(title=f'Validation learning curves · seed {seed}', xlabel='Epoch', ylabel='Normalized MSE')
        ax.legend()
        save_figure(fig, cfg, 'learning_' + names[0])
        plt.show()
        '''),
        ('Interpretation', 'The random router is frozen and input-dependent. The oracle sees true regimes and activates one expert. '
         'A best checkpoint at the last epoch is a reason to examine training length, not proof of convergence. '
         'Do not tune repeatedly against these test errors.', None)])

notebook('05_compare_performance', '05 · Compare prediction and capacity',
    'Compare all five models on the same held-out sequences within each seed. Preserve negative results.', [
    ('Load complete results', 'Run notebooks 03 and 04 first. The table uses original state units squared.', '''
    from twilight_moe.experiment import load_metrics
    from twilight_moe.models import MODEL_NAMES
    results = load_metrics(cfg)
    results = results[results.model.isin(MODEL_NAMES)]
    assert len(results) == len(MODEL_NAMES)*len(cfg['seeds']), 'Train all five models for all seeds first'
    display(results[['model', 'seed', 'test_mse', 'switch_mse', 'rollout_mean_mse',
                     'total_parameters', 'active_parameters', 'trainable_parameters']].round(6))
    '''),
    ('Plot one-step error', 'Bars show seed means. Points show individual seeds, not confidence intervals.', '''
    names = list(MODEL_NAMES)
    means = results.groupby('model')['test_mse'].mean().reindex(names)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(names, means, color=[COLORS[n] for n in names], alpha=.65)
    for i, name in enumerate(names):
        ax.scatter(results.loc[results.model == name, 'test_mse'],
                   np.full(sum(results.model == name), i), color='#222222', marker='x', zorder=3)
    ax.axvline(results['persistence_mse'].mean(), color='#444444', linestyle='--', label='Persistence')
    ax.axvline(results['known_dynamics_mse'].mean(), color='#444444', linestyle=':', label='Known dynamics + true regime')
    ax.set(title=f'One-step held-out error · {cfg["name"]} · {len(cfg["seeds"])} seed(s)',
           xlabel='MSE (original state units squared)', xlim=(0, None))
    ax.legend(fontsize=9)
    save_figure(fig, cfg, 'one_step_comparison')
    plt.show()
    '''),
    ('Calculate paired differences', 'Negative MoE-minus-dense differences favor MoE. Seeds vary both data and initialization.', '''
    paired = results.pivot(index='seed', columns='model', values='test_mse')
    differences = pd.DataFrame({name: paired['moe']-paired[name] for name in ['dense_active', 'dense_total']})
    display(differences.rename(columns=lambda n: 'MoE minus ' + n).round(6))
    print('Per-seed mean differences:', differences.mean().to_dict())
    if len(differences) == 1:
        print('One smoke seed: no uncertainty estimate or general performance conclusion.')
    else:
        print('Sample SD across paired seeds:', differences.std(ddof=1).to_dict())
    '''),
    ('Check capacity matching', 'Report the realized mismatch; active parameter counts do not measure actual runtime.', '''
    capacity = results.groupby('model')[['total_parameters', 'active_parameters']].first()
    display(capacity)
    for dense, field in [('dense_active', 'active_parameters'), ('dense_total', 'total_parameters')]:
        mismatch = (capacity.loc[dense, field]/capacity.loc['moe', field]-1)*100
        print(f'{dense} mismatch: {mismatch:+.2f}%')
    '''),
    ('Takeaways', 'Read the actual paired differences before making a claim. The smoke run verifies mechanics. '
     'Even the full preset is an initial controlled benchmark; broader seeds and equal tuning budgets are needed.', None)])

notebook('06_analyze_routing', '06 · Routing and switch adaptation',
    'Diagnostic questions: are experts used, do routes correspond to regimes, and how do errors change after switches?', [
    ('Load diagnostic results', 'Permutation matching is fitted on validation routes and evaluated on test routes.', '''
    from twilight_moe.experiment import load_metrics, load_diagnostics
    from twilight_moe.models import MODEL_NAMES
    seed = cfg['seeds'][0]
    results = load_metrics(cfg)
    diagnostics = {name: load_diagnostics(cfg, name, seed) for name in MODEL_NAMES}
    display(results.loc[results.model.isin(['moe', 'random', 'oracle']),
                        ['model', 'seed', 'routing_agreement']].round(4))
    '''),
    ('Plot expert utilization', 'Shares count all selected top-k slots. They sum to one, even when k is two.', '''
    fig, ax = plt.subplots()
    for j, name in enumerate(['moe', 'random', 'oracle']):
        ax.bar(np.arange(3)+(j-1)*.25, diagnostics[name]['utilization'], width=.25,
               color=COLORS[name], label=name)
    ax.set(xticks=np.arange(3), xticklabels=['Expert 0', 'Expert 1', 'Expert 2'],
           ylabel='Fraction of selected slots', title=f'Test expert utilization · seed {seed}', ylim=(0, 1))
    ax.legend()
    save_figure(fig, cfg, 'expert_utilization')
    plt.show()
    '''),
    ('Plot regime–expert correspondence', 'Raw test counts use the top-weight selected expert. IDs are arbitrary.', '''
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = diagnostics['moe']['confusion']
    im = ax.imshow(counts, cmap='Blues', vmin=0)
    for row in range(3):
        for col in range(3):
            ax.text(col, row, str(counts[row, col]), ha='center', va='center',
                    color='white' if counts[row, col] > counts.max()*.55 else '#222222')
    ax.set(xticks=range(3), yticks=range(3), xlabel='Top-weight expert', ylabel='True regime',
           title=f'Learned MoE test routing counts · seed {seed}')
    ax.grid(False)
    fig.colorbar(im, ax=ax, label='Prediction windows')
    save_figure(fig, cfg, 'routing_confusion')
    plt.show()
    '''),
    ('Plot switch-aligned error and support', 'At lag zero, a switch may not yet be inferable from the observed history. '
     'Episodes stop at the next switch; support decreases with lag.', '''
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, height_ratios=[3, 1])
    for name in MODEL_NAMES:
        axes[0].plot(diagnostics[name]['adaptation_mse'], color=COLORS[name], label=name, marker='.')
    axes[0].set(title=f'Error after a regime switch · seed {seed}', ylabel='MSE (original units squared)')
    axes[0].legend(ncol=3, fontsize=9)
    axes[1].bar(np.arange(cfg['evaluation']['adaptation_window']), diagnostics['moe']['adaptation_counts'], color='#315b83')
    axes[1].set(xlabel='Steps since switch', ylabel='Episodes')
    save_figure(fig, cfg, 'switch_adaptation')
    plt.show()
    '''),
    ('Inspect recovery and censoring', 'Recovery means three consecutive errors below a model-specific validation threshold '
     '(see research plan). Delays are conditional on recovery and should be read with recovered fractions.', '''
    display(results[['model', 'seed', 'switch_episodes', 'recovered_episodes', 'recovered_fraction',
                     'recovery_delay_steps', 'recovery_threshold']].round(5))
    '''),
    ('Interpretation', 'Routing agreement is a descriptive diagnostic. Good prediction does not require one expert per regime, '
     'and correspondence alone is not evidence of a causal mechanism.', None)])

notebook('07_multi_step_rollout', '07 · Evaluate free-running rollouts',
    'Predict repeatedly by feeding model outputs back into the context. No future observed states are supplied.', [
    ('Load rollout errors', 'Targets do not overlap within each sequence. The oracle is additionally supplied future true regimes.', '''
    from twilight_moe.experiment import load_diagnostics, load_metrics
    from twilight_moe.models import MODEL_NAMES
    results = load_metrics(cfg)
    curves = {name: np.stack([load_diagnostics(cfg, name, seed)['rollout_mse']
                              for seed in cfg['seeds']]) for name in MODEL_NAMES}
    horizon = np.arange(1, cfg['evaluation']['rollout_horizon']+1)
    '''),
    ('Plot errors by forecast horizon', 'Lines are seed means. Light lines show individual seeds when multiple seeds are present.', '''
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in MODEL_NAMES:
        if len(cfg['seeds']) > 1:
            for curve in curves[name]:
                ax.plot(horizon, curve, color=COLORS[name], alpha=.2, linewidth=1)
        ax.plot(horizon, curves[name].mean(0), color=COLORS[name], label=name, marker='.', linewidth=2)
    persistence = np.stack([load_diagnostics(cfg, 'moe', seed)['rollout_persistence'] for seed in cfg['seeds']]).mean(0)
    ax.plot(horizon, persistence, color='#333333', linestyle='--', label='Persistence')
    ax.set(title=f'Free-running rollout error · {cfg["name"]}', xlabel='Forecast horizon (steps)',
           ylabel='MSE (original state units squared)', ylim=(0, None))
    ax.legend(ncol=3, fontsize=9)
    save_figure(fig, cfg, 'rollout_error')
    plt.show()
    '''),
    ('Compare average rollout errors', 'Average error weights each horizon equally. It differs from one-step error over all windows.', '''
    display(results[['model', 'seed', 'rollout_mean_mse', 'rollout_origins']].round(6))
    print('Metrics and diagnostics:', run_dir(cfg))
    print('Figures:', ROOT / 'results' / 'figures' / run_dir(cfg).name)
    '''),
    ('Takeaways and next experiment', 'Long-horizon prediction becomes uncertain as regimes switch and noise accumulates. '
     'An eventually flat error curve can reflect regression toward the mean. '
     'Next, run the larger preset unchanged, compare paired seeds, and then plan a preregistered parameter sweep. '
     'SAE and DAS remain future work.', None)])

print('Created eight notebooks in', ROOT / 'notebooks')
