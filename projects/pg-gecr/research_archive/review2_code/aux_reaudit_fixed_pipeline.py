import numpy as np, json, csv, os, hashlib
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image
from scipy.fftpack import dct
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, matthews_corrcoef, roc_auc_score, classification_report, confusion_matrix, average_precision_score
from sklearn.preprocessing import label_binarize
from scipy.stats import beta

ROOT=Path('/mnt/data/rev2work/auxdata/Otoscopic_Data')
OUT=Path('/mnt/data/rev2work/results'); OUT.mkdir(exist_ok=True)
AUX=Path('/mnt/data/rev2work/aux_learning')
CLASSES=['Acute Otitis Media','Cerumen Impaction','Chronic Otitis Media','Myringosclerosis','Normal']
# Split/arrays produced by documented pHash implementation in aux_learning_curve.py.
z=np.load(AUX/'arrays.npz'); Xtr,ytr,Xv,yv,Xt,yt=[z[k] for k in ['Xtr','ytr','Xv','yv','Xt','yt']]

def features(X):
    # Fixed, transparent region-focused appearance descriptor: masked 64x64 arrays were precomputed.
    # 16x16 regular grid RGB samples + per-channel mean/std.
    return np.concatenate([X[:,:,::4,::4].reshape(len(X),-1), X.mean((2,3)), X.std((2,3))],1).astype(np.float32)
Ftr,Fv,Ft=features(Xtr),features(Xv),features(Xt)

def model():
    return ExtraTreesClassifier(n_estimators=300,max_features=0.5,class_weight='balanced',n_jobs=-1,random_state=20260921)

def calc(y,p,pro):
    return dict(accuracy=float(accuracy_score(y,p)),macro_f1=float(f1_score(y,p,average='macro',zero_division=0)),balanced_accuracy=float(balanced_accuracy_score(y,p)),mcc=float(matthews_corrcoef(y,p)),macro_auc=float(roc_auc_score(y,pro,multi_class='ovr',average='macro')))

# Full fixed pipeline
m=model(); m.fit(Ftr,ytr); pv=m.predict(Fv); pp=m.predict(Ft); pro=m.predict_proba(Ft)
main=calc(yt,pp,pro); main['val_macro_f1']=float(f1_score(yv,pv,average='macro',zero_division=0)); main['n_train']=len(ytr);main['n_val']=len(yv);main['n_test']=len(yt)
# bootstrap image-level 3000
rng=np.random.default_rng(20260921); bs=[]
for b in range(3000):
    ix=rng.integers(0,len(yt),len(yt)); yy=yt[ix]; p=pp[ix]
    # macro F1/accuracy; skip auc bootstrap due missing classes unlikely but possible
    bs.append((accuracy_score(yy,p),f1_score(yy,p,average='macro',zero_division=0),balanced_accuracy_score(yy,p)))
a=np.array(bs)
main['bootstrap95']={'accuracy':np.quantile(a[:,0],[.025,.975]).tolist(),'macro_f1':np.quantile(a[:,1],[.025,.975]).tolist(),'balanced_accuracy':np.quantile(a[:,2],[.025,.975]).tolist()}
json.dump(main,open(OUT/'aux_reaudit_fixed_pipeline_metrics.json','w'),indent=2)
np.savetxt(OUT/'aux_reaudit_fixed_pipeline_confusion.csv',confusion_matrix(yt,pp),delimiter=',',fmt='%d')
# class report + AUC/AP
rep=classification_report(yt,pp,target_names=CLASSES,output_dict=True,zero_division=0)
Y=label_binarize(yt,classes=np.arange(5))
rows=[]
def cp(k,n):
    lo=0 if k==0 else beta.ppf(.025,k,n-k+1); hi=1 if k==n else beta.ppf(.975,k+1,n-k); return float(lo),float(hi)
for c,name in enumerate(CLASSES):
    tp=int(((yt==c)&(pp==c)).sum()); fp=int(((yt!=c)&(pp==c)).sum()); fn=int(((yt==c)&(pp!=c)).sum())
    prec=rep[name]['precision'];rec=rep[name]['recall'];f1=rep[name]['f1-score'];sup=int(rep[name]['support'])
    pci=cp(tp,tp+fp) if tp+fp else (None,None); rci=cp(tp,tp+fn)
    auc=float(roc_auc_score(Y[:,c],pro[:,c])); ap=float(average_precision_score(Y[:,c],pro[:,c]))
    rows.append({'class':name,'tp':tp,'fp':fp,'fn':fn,'precision':prec,'precision_ci_lo':pci[0],'precision_ci_hi':pci[1],'recall':rec,'recall_ci_lo':rci[0],'recall_ci_hi':rci[1],'f1':f1,'auc':auc,'ap':ap,'support':sup})
with open(OUT/'aux_reaudit_fixed_pipeline_classwise.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
# Learning curve same exact model/hyperparams, fixed val/test; balanced subsets only from training.
curves=[]
for t in [25,50,100,200,350]:
    rng=np.random.default_rng(20260921+t); ids=[]
    for c in range(5):
        ix=np.where(ytr==c)[0].copy(); rng.shuffle(ix); ids.extend(ix[:t].tolist())
    ids=np.array(ids); rng.shuffle(ids)
    mm=model();mm.fit(Ftr[ids],ytr[ids]); vv=mm.predict(Fv); p=mm.predict(Ft); pr=mm.predict_proba(Ft)
    r=calc(yt,p,pr); r.update(per_class_training=t,n_train=int(len(ids)),val_macro_f1=float(f1_score(yv,vv,average='macro',zero_division=0)))
    curves.append(r)
with open(OUT/'aux_reaudit_learning_curve.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=curves[0].keys());w.writeheader();w.writerows(curves)
json.dump(curves,open(OUT/'aux_reaudit_learning_curve.json','w'),indent=2)
print(json.dumps(main,indent=2)); print(json.dumps(curves,indent=2)); print(rows)
