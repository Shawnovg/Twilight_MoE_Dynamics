"""Independently reload checkpoints and reconcile all reported one-step errors."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from twilight_moe.config import load_config, run_dir, checkpoint_dir
from twilight_moe.data import load_data
from twilight_moe.evaluation import predict
from twilight_moe.experiment import load_checkpoint, load_metrics, source_hashes, verify_provenance


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/attention_experiment.yaml')
    args=parser.parse_args()
    cfg=load_config(args.config)
    torch.set_num_threads(cfg['training']['threads'])
    assert verify_provenance(cfg)
    rows=load_metrics(cfg)
    checked=[]
    for seed in cfg['seeds']:
        data,training_norm=load_data(cfg,seed)
        for name in cfg['models']:
            model,norm=load_checkpoint(cfg,name,seed)
            saved=torch.load(checkpoint_dir(cfg)/f'{name}_seed{seed}.pt',weights_only=True)
            assert saved['source_files']==source_hashes()
            for key in norm:
                np.testing.assert_array_equal(norm[key],training_norm[key])
            row=rows[(rows.model==name)&(rows.seed==seed)].iloc[0]
            error,_=predict(model,data['test'],cfg['data']['context'],norm)
            difference=abs(float(error.mean())-row.test_mse)
            assert difference<1e-9,(name,seed,difference)
            history=pd.read_csv(run_dir(cfg)/f'{name}_seed{seed}_history.csv')
            assert int(history.loc[history.val_mse_normalized.idxmin(),'epoch'])==int(row.best_epoch)
            assert abs(history.val_mse_normalized.min()-row.best_val_mse_normalized)<1e-12
            checked.append({'model':name,'seed':seed,'mse_difference':difference,'checkpoint_epoch':int(row.best_epoch)})
    result={'config':args.config,'verified_checkpoints':len(checked),'all_source_hashes_match':True,
            'training_normalization_matches':True,'checkpoint_selection_matches_histories':True,'checks':checked}
    (run_dir(cfg)/'checkpoint_verification.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(f'Verified {len(checked)} checkpoints, normalizers, source hashes and selected epochs.')


if __name__=='__main__':
    main()
