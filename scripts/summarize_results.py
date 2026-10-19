"""Write an evidence-backed run log and update the comparison notebook summary."""
import argparse
import sys
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from twilight_moe.config import load_config, run_dir
from twilight_moe.experiment import load_metrics, verify_provenance


def summarize_attention(cfg, config_path):
    verify_provenance(cfg)
    results = load_metrics(cfg)
    paired = results.pivot(index='seed', columns='model', values='test_mse')
    summary = results.groupby('model').test_mse.agg(['mean', 'std']).reindex(cfg['models'])
    rows = [f'# Attention versus FFN routing: {cfg["name"]}', '',
        f'Configuration: `{config_path}`. Seeds: {cfg["seeds"]}. '
        f'{cfg["training"]["epochs"]} epochs per model. '
        'Every model has current per-run source hashes matching the environment manifest.', '',
        '## One-step prediction', '', '| Model | Mean test MSE | Sample SD |', '|---|---:|---:|']
    for name, item in summary.iterrows():
        rows.append(f'| {name} | {item["mean"]:.6f} | {item["std"]:.6f} |')
    rows.extend(['', 'Means weight seeds equally. Each seed uses independent data and initialization. '
                 'SD across three seeds is not a confidence interval.', '', '## Paired comparisons', '',
                 '| First model | Comparator | Mean MSE difference | Mean relative change | First-model wins |',
                 '|---|---|---:|---:|---:|'])
    contrasts = [('attn_moh','attn_dense'), ('attn_moh','attn_total'), ('attn_moh','attn_active'),
        ('attn_moh','attn_random'), ('attn_ffn_moe','attn_dense'), ('attn_ffn_moe','attn_ffn_total'),
        ('attn_both','attn_ffn_moe'), ('attn_both','attn_moh'), ('attn_moh','dense_total'),
        ('attn_moh','moe'), ('attn_dense','dense_total'), ('attn_moh','attn_ffn_moe')]
    for a,b in contrasts:
        delta, relative = paired[a]-paired[b], (paired[a]/paired[b]-1)*100
        rows.append(f'| {a} | {b} | {delta.mean():+.6f} | {relative.mean():+.2f}% | {(delta<0).sum()}/{len(delta)} |')
    interaction = paired.attn_both-paired.attn_moh-paired.attn_ffn_moe+paired.attn_dense
    rows.extend(['', f'2×2 difference-of-differences (both − MoH − FFN-MoE + dense): '
                 f'{interaction.mean():+.6f} MSE. This is descriptive, not a significance test.', '',
                 '## Capacity and training', '',
                 '| Model | Total parameters | Active proxy | Best epochs (seed: epoch) |',
                 '|---|---:|---:|---|'])
    for name in cfg['models']:
        group=results[results.model==name]
        epochs=', '.join(f'{int(r.seed)}: {int(r.best_epoch)}' for r in group.itertuples())
        rows.append(f'| {name} | {int(group.total_parameters.iloc[0])} | '
                    f'{int(group.active_parameters.iloc[0])} | {epochs} |')
    rows.extend(['', 'MoH logically selects 3/4 heads per token, but this reference computes all attention heads. '
                 'Attention active-parameter proxies include all attention parameters. The FFN path dispatches '
                 'only selected expert rows. Neither proxy establishes FLOPs or runtime savings.', '',
                 '## Switch recovery and rollouts', '',
                 '| Model | Switch MSE | Recovered fraction | Conditional delay | Mean rollout MSE |',
                 '|---|---:|---:|---:|---:|'])
    for name in cfg['models']:
        group=results[results.model==name]
        rows.append(f'| {name} | {group.switch_mse.mean():.6f} | {group.recovered_fraction.mean():.3f} | '
                    f'{group.recovery_delay_steps.mean():.3f} | {group.rollout_mean_mse.mean():.6f} |')
    relative=(paired.attn_moh/paired.attn_dense-1)*100
    delta=paired.attn_moh-paired.attn_dense
    finding=(f'MoH has {"lower" if delta.mean()<0 else "higher"} mean one-step error than standard attention: '
             f'{summary.loc["attn_moh","mean"]:.6f} versus {summary.loc["attn_dense","mean"]:.6f}. '
             f'The mean paired relative change is {relative.mean():+.2f}%, with lower error in '
             f'{(delta<0).sum()}/{len(delta)} seeds.')
    control_finding = (f'Against the MoH total-parameter dense control, the mean paired relative change is '
        f'{((paired.attn_moh/paired.attn_total-1)*100).mean():+.2f}%. '
        f'Against the frozen-router control it is '
        f'{((paired.attn_moh/paired.attn_random-1)*100).mean():+.2f}%, with learned MoH winning '
        f'{(paired.attn_moh<paired.attn_random).sum()}/{len(paired)} seeds. '
        'These controls do not establish a specific benefit from learning the head gate.')
    budget_finding = (f'{int((results.best_epoch>=cfg["training"]["epochs"]-2).sum())}/{len(results)} '
        'checkpoints were selected within the last three epochs. This is a fixed-training-budget '
        'comparison, not evidence that every model has reached its best attainable performance.')
    rows.extend(['', '## Interpretation', '', finding, '', control_finding, '', budget_finding, '',
        'A benefit against the old flattened MLP alone would not identify a routing benefit. '
        'The standard-attention, total-parameter and frozen-router controls are essential. '
        'The attention/FFN 2×2 comparison changes capacity as part of adding FFN experts, so read the FFN '
        'total-parameter control too. A comparison of two routing locations is not an isolated test of '
        'one psychological concept of feature gathering versus post-processing.', '',
        'Recovery uses model-specific validation-derived thresholds and conditional delays. '
        'Oracle rollout sees future true regimes. Three seeds, one task, one context length and one training '
        'preset cannot establish general superiority. No test-driven hyperparameter search was performed.', '',
        '## Sources and artifacts', '',
        '- [MoH, Jin et al., ICML 2025](https://proceedings.mlr.press/v267/jin25l.html).',
        '- [MoA, Zhang et al., EMNLP 2022](https://aclanthology.org/2022.emnlp-main.278/).',
        '- Implementation and deviations: `docs/ATTENTION_EXPERIMENT.md`.',
        f'- Metrics, histories, diagnostics and hashes: `{run_dir(cfg).relative_to(ROOT)}`.',
        '- Runnable comparison: notebooks 08–10. Datasets remain on E:.'])
    content='\n'.join(rows)+'\n'
    (run_dir(cfg)/'comparison.md').write_text(content,encoding='utf8')
    if len(cfg['seeds'])>1:
        (ROOT/'results'/'attention_comparison_log.md').write_text(content,encoding='utf8')
        nbpath=ROOT/'notebooks'/'09_compare_attention_and_ffn.ipynb'
        nb=nbformat.read(nbpath,as_version=4)
        nb.cells=[c for c in nb.cells if c.metadata.get('twilight_summary') is not True]
        cell=nbformat.v4.new_markdown_cell('## Saved-run finding\n\n'+finding+
             '\n\nFull preset with three seeds and 40 epochs. This is an initial controlled result. '
             'The default interactive preset is smoke. Recompute before drawing conclusions after changing settings.')
        cell.metadata['twilight_summary']=True
        nb.cells.insert(1,cell)
        nbformat.write(nb,nbpath)
        html,_=HTMLExporter().from_notebook_node(nb)
        (ROOT/'results'/'previews'/'09_compare_attention_and_ffn.html').write_text(html,encoding='utf8')
    print(finding)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/experiment.yaml')
    args = parser.parse_args()
    cfg = load_config(args.config)
    if 'attention' in cfg:
        summarize_attention(cfg, args.config)
        return
    results = load_metrics(cfg)
    summary = results.groupby('model')['test_mse'].agg(['mean', 'std'])
    paired = results.pivot(index='seed', columns='model', values='test_mse')
    differences = {name: paired['moe']-paired[name] for name in ['dense_active', 'dense_total']}
    text = [f'# Experiment log: {cfg["name"]}', '',
        f'Configuration: `{args.config}`. Seeds: {cfg["seeds"]}. '
        f'{cfg["training"]["epochs"]} epochs per model; CPU; validation-selected checkpoints.', '',
        '| Model | Mean test MSE | Sample SD across seeds |', '|---|---:|---:|']
    for name, row in summary.iterrows():
        text.append(f'| {name} | {row["mean"]:.6f} | {row["std"]:.6f} |')
    text.extend(['', 'MSE is averaged over four coordinates in original state units squared. '
        'Seed spread is not a confidence interval. Seeds vary both dataset and initialization.', '',
        '## Paired comparisons', ''])
    statements = []
    for name, delta in differences.items():
        relative = (paired['moe']/paired[name]-1)*100
        statement = (f'Against {name}, learned MoE mean paired MSE difference is {delta.mean():+.6f} '
            f'(sample SD {delta.std(ddof=1):.6f}); mean within-seed relative change '
            f'{relative.mean():+.2f}%. MoE has lower error in {(delta < 0).sum()}/{len(delta)} seeds.')
        statements.append(statement)
        text.append('- ' + statement)
    text.extend(['', '## Capacity and checkpoint selection', '',
        '| Model | Total parameters | Active parameter proxy | Selected epochs by seed |',
        '|---|---:|---:|---|'])
    for name, group in results.groupby('model'):
        row = group.iloc[0]
        epochs = ', '.join(f'{int(r.seed)}: {int(r.best_epoch)}' for r in group.itertuples())
        text.append(f'| {name} | {int(row.total_parameters)} | {int(row.active_parameters)} | {epochs} |')
    text.extend(['', '## Interpretation and limits', '',
        f'Persistence MSE averaged over seeds: {results.persistence_mse.mean():.6f}. '
        f'Known dynamics with true regime: {results.known_dynamics_mse.mean():.6f}; '
        f'configured noise variance: {cfg["data"]["noise_std"]**2:.6f}.', '',
        'The trained oracle receives privileged regime labels. Its error is a reference, '
        'not a guaranteed optimization bound. Parameter proxies do not establish runtime or FLOP savings. '
        'Random routing uses a frozen linear gate, with its smaller router included in capacity reporting.', '',
        'This is one synthetic task and one fixed hyperparameter preset. Three seeds do not establish '
        'a general advantage or disadvantage for MoE. Report the result as an initial controlled benchmark; '
        'test broader seeds, context lengths, switching rates, and equal tuning budgets before a larger claim.', '',
        'Switch recovery delays are conditional on recovery; unrecovered episodes are censored. '
        'Thresholds are model-specific and validation-derived. Read delays together with recovery fractions. '
        'Multi-step oracle rollouts receive future true regimes.', '',
        '## Reproduction and artifacts', '',
        f'- Metrics and provenance: `{run_dir(cfg).relative_to(ROOT)}`.',
        f'- Figures: `results/figures/{run_dir(cfg).name}`.',
        '- Dataset source and checksums: the configured E: data root, `metadata/`.',
        '- Exact installed package versions: `requirements-lock.txt`.',
        '- Saved notebook outputs were produced by `scripts/execute_notebooks.py`.',
        '- Smoke results are stored separately from the full preset.',
        '- Unit test and visual verification outcomes are recorded in `VERIFICATION.md`.'])
    (ROOT/'results'/'experiment_log.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    nbpath = ROOT/'notebooks'/'05_compare_performance.ipynb'
    nb = nbformat.read(nbpath, as_version=4)
    nb.cells = [c for c in nb.cells if c.metadata.get('twilight_summary') is not True]
    summary_cell = nbformat.v4.new_markdown_cell('## Saved-run findings\n\n'
        f'These saved outputs use **{cfg["name"]}**, {len(cfg["seeds"])} seeds, '
        f'and {cfg["training"]["epochs"]} epochs.\n\n' + '\n\n'.join(statements) +
        '\n\nThis is an initial controlled benchmark, not a general claim. '
        '**Rerunning without `TWILIGHT_CONFIG` uses the smoke preset.** '
        'This saved summary is regenerated with `scripts/summarize_results.py`; '
        'after changing settings, use the new computed tables below as the current results.')
    summary_cell.metadata['twilight_summary'] = True
    nb.cells.insert(1, summary_cell)
    nbformat.write(nb, nbpath)
    html, _ = HTMLExporter().from_notebook_node(nb)
    (ROOT/'results'/'previews'/'05_compare_performance.html').write_text(html, encoding='utf-8')
    print('\n'.join(statements))


if __name__ == '__main__':
    main()
