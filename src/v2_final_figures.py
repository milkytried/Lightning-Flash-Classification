from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()


def save_bar(labels, values, title, ylabel, path):
    fig, ax = plt.subplots(figsize=(7,4))
    bars = ax.bar(labels, values, color=['#4c78a8','#f58518','#54a24b','#b279a2'][:len(labels)])
    ax.set_title(title); ax.set_ylabel(ylabel); ax.set_ylim(0, max(values)*1.15 if values else 1)
    ax.tick_params(axis='x', rotation=20)
    for bar, val in zip(bars, values): ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--out', default='results/v2/final_figures'); args=parser.parse_args()
    out=Path(args.out); out.mkdir(parents=True, exist_ok=True)
    final=json.load(open('report/V2_PHASE3_FINAL_DECISION.json'))
    controlled=json.load(open('report/V2_PHASE3_CONTROLLED_TEST.json'))
    natural=json.load(open('report/V2_PHASE3_NATURAL_PREVALENCE.json'))
    training=json.load(open('report/V2_PHASE3_TRAINING.json'))
    baselines=json.load(open('report/V2_FULL_BASELINES.json'))
    selected='small_cnn_seed2026_bce_pos_weight_train_split_none'
    c_pred=pd.read_csv(f'results/v2/phase3/controlled_test_predictions/{selected}.csv')
    n_pred=pd.read_csv(f'results/v2/phase3/natural_prevalence_predictions/{selected}.csv')
    artifacts=[]
    def add(path, caption):
        artifacts.append({'path':str(path),'caption':caption,'sha256':sha256_file(path),'bytes':path.stat().st_size})
    # V1/V2 comparison
    path=out/'version1_vs_version2_comparison.png'
    fig, ax=plt.subplots(figsize=(7,4)); labels=['V1 diagnostic\nROC-AUC','V2 controlled\nROC-AUC','V2 natural\nROC-AUC']; vals=[0.9681, final['best_neural_controlled_test']['roc_auc'], next(x for x in natural['results'] if x['run_name']==selected)['metrics_at_validation_threshold']['roc_auc']]
    ax.bar(labels, vals, color=['#9ecae1','#31a354','#74c476']); ax.set_ylim(0,1); ax.set_ylabel('ROC-AUC'); ax.set_title('Version 1 diagnostic vs Version 2 frozen result')
    for i,v in enumerate(vals): ax.text(i,v+0.01,f'{v:.3f}',ha='center')
    fig.tight_layout(); fig.savefig(path,dpi=300); plt.close(fig); add(path,'Version 1 retained as diagnostic benchmark; Version 2 is the corrected scientific experiment.')
    # validation seed bars
    rows=training['runs']; path=out/'validation_pr_auc_by_seed.png'; fig, ax=plt.subplots(figsize=(8,4)); labs=[r['run_name'].replace('_bce_pos_weight_train_split_none','').replace('_bce_unweighted_none','') for r in rows]; vals=[r['validation_pr_auc'] for r in rows]
    ax.bar(labs, vals, color=['#31a354']*3+['#756bb1']*3); ax.set_ylim(0.75,1.0); ax.set_ylabel('Validation PR-AUC'); ax.set_title('Validation PR-AUC across frozen primary seeds'); ax.tick_params(axis='x', rotation=35)
    fig.tight_layout(); fig.savefig(path,dpi=300); plt.close(fig); add(path,'Validation-only seed stability for selected small CNN and frozen ResNet-50 configurations.')
    # confusion matrices
    for split, pred, title in [('controlled',c_pred,'Controlled test'),('natural',n_pred,'Natural-prevalence test')]:
        thr=float(pred.threshold.iloc[0]); cm=confusion_matrix(pred.label, pred.probability>=thr, labels=[0,1]); path=out/f'{split}_selected_confusion_matrix.png'; fig, ax=plt.subplots(figsize=(4,3.6)); im=ax.imshow(cm,cmap='Blues'); ax.set_xticks([0,1],['No lightning','Lightning'],rotation=20,ha='right'); ax.set_yticks([0,1],['No lightning','Lightning']); ax.set_xlabel('Predicted'); ax.set_ylabel('Observed'); ax.set_title(f'{title}: selected small CNN')
        for i in range(2):
            for j in range(2): ax.text(j,i,str(cm[i,j]),ha='center',va='center')
        fig.colorbar(im,ax=ax,fraction=.046,pad=.04); fig.tight_layout(); fig.savefig(path,dpi=300); plt.close(fig); add(path,f'{title} confusion matrix for the final selected small CNN.')
    # roc/pr selected
    for split,pred in [('controlled',c_pred),('natural',n_pred)]:
        for kind in ['roc','precision_recall']:
            path=out/f'{split}_{kind}_selected.png'; fig, ax=plt.subplots(figsize=(5,4))
            if kind=='roc': RocCurveDisplay.from_predictions(pred.label,pred.probability,ax=ax,name='Small CNN'); ax.plot([0,1],[0,1],'k--',lw=1); cap='ROC curve'
            else: PrecisionRecallDisplay.from_predictions(pred.label,pred.probability,ax=ax,name='Small CNN'); cap='Precision-recall curve'
            ax.set_title(f'{split.replace("_"," ").title()} {cap}'); fig.tight_layout(); fig.savefig(path,dpi=300); plt.close(fig); add(path,f'{cap} for selected model on {split} split.')
    # reliability/probability hist
    for split,pred in [('controlled',c_pred),('natural',n_pred)]:
        path=out/f'{split}_probability_distribution_selected.png'; fig,ax=plt.subplots(figsize=(5,4)); ax.hist(pred.loc[pred.label.eq(0),'probability'],bins=30,alpha=.65,label='No lightning'); ax.hist(pred.loc[pred.label.eq(1),'probability'],bins=30,alpha=.65,label='Lightning'); ax.axvline(float(pred.threshold.iloc[0]),color='black',ls='--',label='Threshold'); ax.set_title(f'{split.title()} probability distributions'); ax.set_xlabel('Probability'); ax.set_ylabel('Count'); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=300); plt.close(fig); add(path,f'Probability distributions by true label on {split}.')
    # baseline comparison
    path=out/'baseline_comparison_pr_auc.png'; names=['Geo/time RF','B13-min RF','Small CNN']; vals=[baselines['latlon_time_month_random_forest']['test']['pr_auc'], baselines['b13_min_random_forest']['test']['pr_auc'], final['best_neural_controlled_test']['pr_auc']]; save_bar(names, vals, 'Controlled-test PR-AUC baseline comparison','PR-AUC',path); add(path,'Small CNN materially exceeds geographic/time and B13-minimum baselines on controlled PR-AUC.')
    # subgroup bars natural
    best_nat=next(x for x in natural['results'] if x['run_name']==selected)
    for col,label in [('frame_category','Active vs zero-recorded frames'),('date','Per-date accuracy (first 20 dates)')]:
        vals=best_nat['subgroups'].get(col,{})
        items=list(vals.items())[:20]
        if items:
            path=out/f'natural_{col}_subgroup_accuracy.png'; save_bar([k for k,v in items],[v['accuracy'] for k,v in items],label,'Accuracy',path); add(path,f'Natural-prevalence subgroup accuracy by {col}.')
    # month/local from prediction metadata
    n_pred['timestamp']=pd.to_datetime(n_pred['date'])
    n_pred['month']=n_pred['timestamp'].dt.month.astype(str)
    if 'frame_category' in n_pred:
        for col in ['month','frame_category']:
            vals=[]; labs=[]; thr=float(n_pred.threshold.iloc[0])
            for key,part in n_pred.groupby(col):
                labs.append(str(key)); vals.append(float(((part.probability>=thr).astype(int)==part.label).mean()))
            path=out/f'natural_{col}_prediction_accuracy.png'; save_bar(labs,vals,f'Natural-prevalence accuracy by {col}','Accuracy',path); add(path,f'Natural-prevalence selected-model accuracy grouped by {col}.')
    # example TP/TN/FP/FN patches
    examples=[]; thr=float(c_pred.threshold.iloc[0]); c_pred['pred']=(c_pred.probability>=thr).astype(int)
    for label,name in [((1,1),'TP'),((0,0),'TN'),((0,1),'FP'),((1,0),'FN')]:
        part=c_pred[(c_pred.label==label[0])&(c_pred.pred==label[1])]
        if len(part): examples.append((name,part.iloc[0].path))
    if examples:
        path=out/'example_tp_tn_fp_fn_patches.png'; fig,axs=plt.subplots(1,len(examples),figsize=(3*len(examples),3));
        if len(examples)==1: axs=[axs]
        for ax,(name,p) in zip(axs,examples): ax.imshow(Image.open(p)); ax.set_title(name); ax.axis('off')
        fig.suptitle('Selected-model controlled-test examples'); fig.tight_layout(); fig.savefig(path,dpi=300); plt.close(fig); add(path,'Example controlled-test TP, TN, FP and FN patches for the selected model.')
    # provenance
    prov={'generated_at_utc':datetime.now(timezone.utc).isoformat(), 'command':'python src/v2_final_figures.py --out results/v2/final_figures', 'source_commit':git_commit(), 'inputs':{p:sha256_file(p) for p in ['report/V2_PHASE3_FINAL_DECISION.json','report/V2_PHASE3_CONTROLLED_TEST.json','report/V2_PHASE3_NATURAL_PREVALENCE.json','report/V2_PHASE3_TRAINING.json','report/V2_FULL_BASELINES.json',f'results/v2/phase3/controlled_test_predictions/{selected}.csv',f'results/v2/phase3/natural_prevalence_predictions/{selected}.csv']}, 'artifacts':artifacts}
    (out/'figure_provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
    md=['# Version 2 Figure Provenance','',f"Generated: `{prov['generated_at_utc']}`",'',f"Command: `{prov['command']}`",'',f"Source commit: `{prov['source_commit']}`",'', '## Inputs','']
    md += [f"- `{k}`: `{v}`" for k,v in prov['inputs'].items()]
    md += ['', '## Output Figures',''] + [f"- `{a['path']}` ({a['bytes']} bytes): `{a['sha256']}` — {a['caption']}" for a in artifacts]
    Path('docs/version2_figure_provenance.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(json.dumps({'figures':len(artifacts),'provenance':'docs/version2_figure_provenance.md'},indent=2))

if __name__=='__main__': main()
