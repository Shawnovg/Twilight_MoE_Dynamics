"""Create the attention comparison notebooks without rewriting notebooks 00--07."""
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
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from twilight_moe.config import load_config, run_dir
from twilight_moe.experiment import load_metrics, load_diagnostics, verify_provenance
from twilight_moe.attention import ATTENTION_NAMES
from twilight_moe.plotting import style, save_figure, COLORS
config_path = os.environ.get('TWILIGHT_CONFIG', 'configs/attention_smoke.yaml')
cfg = load_config(config_path)
if 'attention' not in cfg:
    cfg = load_config('configs/attention_smoke.yaml')
style()
print('Preset:', cfg['name'], '| seeds:', cfg['seeds'], '| epochs:', cfg['training']['epochs'])
print('Data:', cfg['data_root'])
'''


def write(name, title, intro, sections):
    cells = [nbf.v4.new_markdown_cell('# '+title+'\n\n'+intro),
             nbf.v4.new_markdown_cell('## Setup\n\nRun cells in order. Default: smoke. '
                 'Set `TWILIGHT_CONFIG=configs/attention_experiment.yaml` for the full preset.'),
             nbf.v4.new_code_cell(dedent(SETUP).strip())]
    for title, body, code in sections:
        cells.append(nbf.v4.new_markdown_cell('## '+title+'\n\n'+body))
        if code:
            cells.append(nbf.v4.new_code_cell(dedent(code).strip()))
    nb = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3 (ipykernel)', 'language': 'python', 'name': 'python3'}})
    nbf.validate(nb)
    nbf.write(nb, ROOT/'notebooks'/name)


write('08_train_attention_comparison.ipynb', '08 · Attention versus FFN routing',
      'A controlled extension of the original switching-dynamics experiment. '
      'Full study: 13 model families, three paired seeds, unchanged data and 40 epochs. '
      'Smoke execution checks mechanics only.', [
    ('Context and references',
     '[MoH: Multi-Head Attention as Mixture-of-Head Attention](https://proceedings.mlr.press/v267/jin25l.html) '
     '(Jin et al., ICML 2025) routes tokens between shared and selected heads. '
     '[Mixture of Attention Heads: Selecting Attention Heads Per Token](https://aclanthology.org/2022.emnlp-main.278/) '
     '(Zhang et al., EMNLP 2022) is an earlier token-level head-routing design. '
     'Our experiment implements a small MoH-inspired predictor, not either paper’s benchmark reproduction. '
     'See `docs/ATTENTION_EXPERIMENT.md` for exact mechanisms and deviations.', None),
    ('Data and design', 'Same 4-D process, split sequences, training-only normalization and 8-state history. '
     'Regime labels enter only the original oracle and post-hoc diagnostics.', '''
    display(pd.Series(cfg['data'], name='Data setting').to_frame())
    display(pd.Series(cfg['attention'], name='Attention setting').to_frame())
    '''),
    ('Architecture and capacities', 'Two causal pre-norm blocks, token embedding plus learned positions, '
     'and final-token readout. MoH uses one shared head and top-2 of three routed heads. '
     'Attention computations remain dense in this reference. Active parameter proxies do not mean FLOPs.', '''
    from twilight_moe.models import build_model
    capacities = pd.DataFrame([{'model': n, **build_model(n, cfg).capacity()} for n in cfg['models']])
    display(capacities.fillna('—'))
    print(build_model('attn_moh', cfg))
    '''),
    ('Train or validate a complete saved run', 'This cell reuses only complete runs whose per-family and '
     'manifest source hashes match. Set FORCE_RETRAIN=True to train all families again. '
     'Original MLP baselines rerun in the same experiment namespace.', '''
    from twilight_moe.experiment import run_models
    FORCE_RETRAIN = False
    try:
        valid_cache = verify_provenance(cfg)
    except (FileNotFoundError, ValueError, KeyError):
        valid_cache = False
    if FORCE_RETRAIN or not valid_cache:
        run_models(cfg)
    assert verify_provenance(cfg)
    results = load_metrics(cfg)
    display(results[['model','seed','best_epoch','test_mse','training_seconds']].round(6))
    '''),
    ('Validation trajectories', 'First seed only. Pure validation prediction MSE determines checkpoints. '
     'All models use the same minibatch order for each seed.', '''
    selected = ['attn_dense','attn_moh','attn_ffn_moe','attn_both']
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, name in enumerate(selected):
        history = pd.read_csv(run_dir(cfg)/f"{name}_seed{cfg['seeds'][0]}_history.csv")
        ax.plot(history.epoch, history.val_mse_normalized, color=COLORS[name],
                linestyle=['-','--','-.',':'][i], label=name)
    ax.set(xlabel='Epoch', ylabel='Normalized validation MSE', title='Validation learning curves')
    ax.legend()
    save_figure(fig, cfg, 'attention_learning_curves')
    plt.show()
    ''')])

write('09_compare_attention_and_ffn.ipynb', '09 · Compare attention and FFN routing',
      'Read the paired contrasts before interpreting a ranking. Dense attention controls distinguish '
      'the benefit of attention itself from the benefit of head routing.', [
    ('Load reviewed results', 'Source hashes must match every family and seed. '
     'The full preset is an initial benchmark, not a general verdict on MoH.', '''
    assert verify_provenance(cfg)
    results = load_metrics(cfg)
    paired = results.pivot(index='seed', columns='model', values='test_mse')
    summary = results.groupby('model')['test_mse'].agg(['mean','std']).reindex(cfg['models'])
    display(summary.round(6))
    '''),
    ('One-step comparison', 'Bars show seed means. Individual markers show seed variation, not confidence intervals. '
     'The oracle has privileged labels.', '''
    fig, ax = plt.subplots(figsize=(11, 7))
    names = cfg['models']
    ax.barh(names, summary['mean'], color=[COLORS[n] for n in names], alpha=.7)
    for i, name in enumerate(names):
        ax.scatter(paired[name], np.full(len(paired), i), marker='x', color='#222222', zorder=3)
    ax.axvline(results.known_dynamics_mse.mean(), color='#444444', linestyle=':', label='Known dynamics + true regime')
    ax.set(xlabel='Test MSE (original state units squared)', title='One-step prediction across model families', xlim=(0,None))
    ax.invert_yaxis()
    ax.legend()
    save_figure(fig, cfg, 'attention_one_step_comparison')
    plt.show()
    '''),
    ('Paired routing contrasts', 'Negative differences favor the first model. '
     'Capacity controls are separate from the common-backbone 2×2 comparison.', '''
    contrasts = [('attn_moh','attn_dense'), ('attn_moh','attn_total'),
                 ('attn_moh','attn_active'), ('attn_moh','attn_random'),
                 ('attn_ffn_moe','attn_dense'), ('attn_ffn_moe','attn_ffn_total'),
                 ('attn_both','attn_ffn_moe'), ('attn_both','attn_moh')]
    deltas = pd.DataFrame({f'{a} minus {b}':paired[a]-paired[b] for a,b in contrasts})
    display(deltas.T.round(6))
    fig, ax = plt.subplots(figsize=(11, 6))
    for j, (seed, row) in enumerate(deltas.iterrows()):
        ax.scatter(row.values, np.arange(len(row))+(j-(len(deltas)-1)/2)*.13,
                   label=f'Seed {seed}', marker=['o','x','s'][j%3], color=['#315b83','#b47719','#965f86'][j%3])
    ax.axvline(0, color='#444444', linewidth=1)
    ax.set(yticks=np.arange(len(deltas.columns)), yticklabels=deltas.columns,
           xlabel='Paired test MSE difference', title='Within-backbone and control comparisons')
    ax.invert_yaxis()
    ax.legend()
    save_figure(fig, cfg, 'attention_paired_contrasts')
    plt.show()
    '''),
    ('Switch adaptation', 'Average seed-level curves. Each episode stops at the next switch, so later lags '
     'have fewer eligible episodes. The lower panel shows support.', '''
    selected = ['attn_dense','attn_moh','attn_ffn_moe','attn_both']
    diagnostics = {n:[load_diagnostics(cfg,n,seed) for seed in cfg['seeds']] for n in selected}
    fig, axes = plt.subplots(2,1,figsize=(10,6),sharex=True,height_ratios=[3,1])
    for i, name in enumerate(selected):
        curve = np.mean([x['adaptation_mse'] for x in diagnostics[name]], axis=0)
        axes[0].plot(curve, label=name, color=COLORS[name], linestyle=['-','--','-.',':'][i])
    counts = np.sum([x['adaptation_counts'] for x in diagnostics['attn_dense']], axis=0)
    axes[0].set(title='Prediction error after regime switches', ylabel='MSE (original units squared)')
    axes[0].legend()
    axes[1].bar(np.arange(len(counts)),counts,color='#315b83')
    axes[1].set(xlabel='Steps since true switch',ylabel='Episodes')
    save_figure(fig,cfg,'attention_switch_adaptation')
    plt.show()
    '''),
    ('Recursive forecasts', 'No future observed states enter ordinary models. '
     'Oracle and persistence remain available in the full metric files.', '''
    fig, ax = plt.subplots(figsize=(10,5))
    for i,name in enumerate(selected):
        curve = np.mean([x['rollout_mse'] for x in diagnostics[name]],axis=0)
        ax.plot(np.arange(1,len(curve)+1),curve,label=name,color=COLORS[name],linestyle=['-','--','-.',':'][i])
    ax.set(xlabel='Forecast horizon',ylabel='MSE (original units squared)',title='Recursive forecast error')
    ax.legend()
    save_figure(fig,cfg,'attention_rollout_comparison')
    plt.show()
    display(results.groupby('model')[['switch_mse','recovered_fraction','recovery_delay_steps','rollout_mean_mse']].mean().round(6))
    '''),
    ('Interpretation', 'The comparison tests a fixed configuration without test-driven tuning. '
     'A routing win does not show that the dataset requires attention, and a loss does not invalidate MoH. '
     'Three seeds change both data and initialization. Read the generated attention comparison log for the measured conclusion.', None)])

write('10_inspect_attention_routing.ipynb', '10 · Attention patterns and head routing',
      'These diagnostics describe route use and attention weights. They do not identify causal explanations.', [
    ('Load diagnostics', 'All examples and routes use the held-out test split after validation checkpoint selection.', '''
    assert verify_provenance(cfg)
    seed = cfg['seeds'][0]
    diagnostic = load_diagnostics(cfg,'attn_moh',seed)
    display(load_metrics(cfg).query("model in ['attn_moh','attn_random','attn_both']")
            [['model','seed','head_layer0_agreement','head_layer1_agreement']].round(4))
    '''),
    ('Routed-head utilization', 'Final-query routing only. Shared head 0 is always active and excluded here. '
     'Slot shares sum to one. Expert indices are arbitrary.', '''
    fig, axes = plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    for layer,ax in enumerate(axes):
        for j,name in enumerate(['attn_moh','attn_random']):
            values=load_diagnostics(cfg,name,seed)[f'head_layer{layer}_utilization']
            ax.bar(np.arange(3)+(j-.5)*.32,values,width=.32,color=COLORS[name],label=name)
        ax.set(xticks=range(3),xticklabels=['Head 1','Head 2','Head 3'],ylim=(0,1),
               title=f'Block {layer+1}',ylabel='Fraction of selected routed slots')
        ax.legend()
    fig.suptitle(f'Routed attention-head utilization · seed {seed}')
    save_figure(fig,cfg,'attention_head_utilization')
    plt.show()
    '''),
    ('Attention over history', 'Mean ungated attention probabilities for the final query. '
     'The graph includes all heads, even for tokens where a routed head was inactive. '
     'Position 0 is the oldest state and 7 the most recent.', '''
    patterns=diagnostic['last_query_attention_mean']
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for layer,ax in enumerate(axes):
        im=ax.imshow(patterns[layer],vmin=0,vmax=1,cmap='Blues',aspect='auto')
        ax.set(xlabel='History position',yticks=range(4),yticklabels=['Shared 0','Routed 1','Routed 2','Routed 3'],
               xticks=range(cfg['data']['context']),title=f'Block {layer+1}')
        ax.grid(False)
    fig.colorbar(im,ax=axes,label='Mean attention probability')
    save_figure(fig,cfg,'attention_history_patterns')
    plt.show()
    '''),
    ('Regime association', 'Top routed head against the current transition regime, final query only. '
     'A validation-fitted permutation produces the reported test agreement. '
     'The first block gate sees only the current token embedding and position.', '''
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for layer,ax in enumerate(axes):
        counts=diagnostic[f'head_layer{layer}_confusion']
        im=ax.imshow(counts,cmap='Blues',vmin=0)
        for row in range(3):
            for col in range(3):
                ax.text(col,row,str(counts[row,col]),ha='center',va='center',
                        color='white' if counts[row,col]>counts.max()*.55 else '#222222')
        ax.set(xticks=range(3),xticklabels=['Head 1','Head 2','Head 3'],yticks=range(3),
               xlabel='Top routed head',ylabel='True regime',title=f'Block {layer+1}')
        ax.grid(False)
        fig.colorbar(im,ax=ax,label='Test windows')
    save_figure(fig,cfg,'attention_regime_confusion')
    plt.show()
    '''),
    ('Head weights', 'Averaged across all tokens and test windows. Shared/routed type weights are learned. '
     'Under the paper equation, masking full-softmax weights can leave total retained mass below one.', '''
    display(pd.DataFrame(diagnostic['head_weight_mean'],index=['Block 1','Block 2'],
                         columns=['Shared 0','Routed 1','Routed 2','Routed 3']).round(4))
    ''')])

# Keep the old comparison focused on its five original model families.
path=ROOT/'notebooks'/'05_compare_performance.ipynb'
nb=nbf.read(path,as_version=4)
for cell in nb.cells:
    if cell.cell_type=='code' and 'results = load_metrics(cfg)' in cell.source and 'assert len(results)' in cell.source:
        if 'results = results[results.model.isin(MODEL_NAMES)]' not in cell.source:
            cell.source=cell.source.replace('results = load_metrics(cfg)',
                'results = load_metrics(cfg)\nresults = results[results.model.isin(MODEL_NAMES)]')
nbf.write(nb,path)
print('Created notebooks 08--10; updated model filtering in notebook 05.')
