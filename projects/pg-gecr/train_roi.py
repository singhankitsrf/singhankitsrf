"""Train/export the locked expert-only ROI configuration on supplied features.

New runs are not recovered original model checkpoints or paper replication.
"""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, matthews_corrcoef
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent


def temperature_scale(p, temperature):
    z = np.log(np.clip(p, 1e-9, 1)) / temperature
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train(features, output, config=HERE/'configs/study.json'):
    cfg = json.loads(config.read_text())
    n_classes = len(cfg['class_order'])
    with np.load(features, allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in ('Xtr','ytr','Xv','yv','Xt','yt')}
    dimension = arrays['Xtr'].shape[1] if arrays['Xtr'].ndim == 2 else 0
    for suffix in ('tr','v','t'):
        x, y = arrays['X'+suffix], arrays['y'+suffix]
        if x.ndim != 2 or x.shape[1] != dimension or not np.isfinite(x).all():
            raise ValueError('Feature matrices must be finite and have matching dimensions')
        if y.ndim != 1 or len(y) != len(x) or not np.issubdtype(y.dtype, np.integer):
            raise ValueError('Labels must be integer vectors matching feature rows')
        if len(y) == 0 or np.any(y < 0) or np.any(y >= n_classes):
            raise ValueError('Invalid or empty labels')
    if set(arrays['ytr']) != set(range(n_classes)):
        raise ValueError('Training must include all four classes')
    m = cfg['primary_model']
    model = Pipeline([('s', StandardScaler()),
                      ('p', PCA(n_components=m['pca_n_components'], whiten=m['pca_whiten'], random_state=cfg['seed'])),
                      ('m', LogisticRegression(C=m['logistic_C'], max_iter=m['max_iter'],
                                               class_weight=m['class_weight'], random_state=cfg['seed']))])
    model.fit(arrays['Xtr'], arrays['ytr'])
    pv = model.predict_proba(arrays['Xv'])
    temperature = float(minimize_scalar(
        lambda t: log_loss(arrays['yv'], temperature_scale(pv,t), labels=np.arange(n_classes)),
        bounds=tuple(cfg['calibration']['temperature_bounds']), method='bounded').x)
    # The test partition is only evaluated after fitting and calibration are locked.
    pt = temperature_scale(model.predict_proba(arrays['Xt']), temperature)
    predicted, truth = pt.argmax(1), arrays['yt']
    metrics = {'accuracy': float(accuracy_score(truth,predicted)),
               'balanced_accuracy': float(balanced_accuracy_score(truth,predicted)),
               'macro_f1': float(f1_score(truth,predicted,labels=np.arange(n_classes),average='macro',zero_division=0)),
               'mcc': float(matthews_corrcoef(truth,predicted)),
               'log_loss': float(log_loss(truth,pt,labels=np.arange(n_classes)))}
    output.mkdir(parents=True,exist_ok=True)
    joblib.dump(model, output/'expert_roi.joblib')
    np.savez_compressed(output/'probabilities.npz', y=truth,probs=pt,yval=arrays['yv'],
                        pval=temperature_scale(pv,temperature),temperature=temperature)
    report = {'scope':'new_training_run_on_user_supplied_features',
              'original_paper_weights':False,'class_order':cfg['class_order'],
              'temperature':temperature,'feature_dimension':dimension,
              'split_counts':{s:len(arrays[k]) for s,k in [('train','ytr'),('validation','yv'),('test','yt')]},
              'input_sha256':hashlib.sha256(features.read_bytes()).hexdigest(),
              'config':cfg,'test_metrics':metrics}
    (output/'model.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'SHA256SUMS').write_text(''.join(hashlib.sha256((output/n).read_bytes()).hexdigest()+'  '+n+'\n'
                                           for n in ('expert_roi.joblib','probabilities.npz','model.json')))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=HERE/'configs/study.json')
    a = p.parse_args()
    try: print(json.dumps(train(a.features,a.output,a.config),indent=2))
    except (ValueError,KeyError,OSError) as e: p.error(str(e))
