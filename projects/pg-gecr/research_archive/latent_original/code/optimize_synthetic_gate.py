import numpy as np, pandas as pd, json
from pathlib import Path
from itertools import product
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, matthews_corrcoef, roc_auc_score, log_loss
ROOT=Path('/mnt/data/eswa_genai_project/experiment_outputs')
a=np.load(ROOT/'cvae_hypothesis_embeddings.npz'); s=np.load(ROOT/'cvae_synthetic_embeddings.npz')
Xtr,Xv,Xt,Xp=a['Xtr'],a['Xv'],a['Xt'],a['Xp']; ytr,yv,yt,yp=a['ytr'],a['yv'],a['yt'],a['yp']; Xs,ys=s['Xs'],s['ys']

def met(y,p):
 pr=p.argmax(1); pre,rec,f1,_=precision_recall_fscore_support(y,pr,average='macro',zero_division=0); auc=roc_auc_score(label_binarize(y,classes=np.arange(4)),p,multi_class='ovr',average='macro')
 return dict(accuracy=accuracy_score(y,pr),balanced_accuracy=balanced_accuracy_score(y,pr),macro_precision=pre,macro_recall=rec,macro_f1=f1,macro_auc=auc,MCC=matthews_corrcoef(y,pr),log_loss=log_loss(y,p,labels=np.arange(4)))
# rank synth by standardized Euclidean distance to expert class centroid
rank={}
for c in [0,1,2]:
 ex=Xtr[ytr==c]; mu=ex.mean(0); sd=ex.std(0)+1e-3; ids=np.where(ys==c)[0]; d=np.mean(((Xs[ids]-mu)/sd)**2,axis=1); rank[c]=ids[np.argsort(d)]

models={
 'Logistic':lambda:Pipeline([('s',StandardScaler()),('m',LogisticRegression(C=.1,max_iter=3000,class_weight='balanced',random_state=42))]),
 'ExtraTrees':lambda:ExtraTreesClassifier(n_estimators=500,max_features='sqrt',class_weight='balanced',min_samples_leaf=2,random_state=42,n_jobs=-1),
 'MLP':lambda:Pipeline([('s',StandardScaler()),('m',MLPClassifier(hidden_layer_sizes=(96,48),alpha=.001,early_stopping=True,max_iter=700,random_state=42))])
}
baseX=np.vstack([Xtr,Xp]); basey=np.r_[ytr,yp]
fracs=[0,.1,.25,.5,.75,1.0]
rows=[]; best=None
# same fraction all classes plus targeted combos coarse
configs=[(f,f,f) for f in fracs]
configs += list(product([0,.25,.5,1.0], repeat=3))
configs=sorted(set(configs))
for f0,f1,f2 in configs:
 ids=np.r_[rank[0][:round(len(rank[0])*f0)],rank[1][:round(len(rank[1])*f1)],rank[2][:round(len(rank[2])*f2)]] if any([f0,f1,f2]) else np.array([],dtype=int)
 XX=np.vstack([baseX,Xs[ids]]) if len(ids) else baseX; yy=np.r_[basey,ys[ids]] if len(ids) else basey
 for name,fn in models.items():
  m=fn();m.fit(XX,yy);pv=m.predict_proba(Xv);pt=m.predict_proba(Xt);vm=met(yv,pv);tm=met(yt,pt)
  row={'model':name,'f_AOM':f0,'f_ASOM':f1,'f_CSOM':f2,'n_synth':len(ids),**{'val_'+k:v for k,v in vm.items()},**{'test_'+k:v for k,v in tm.items()}}
  rows.append(row)
  if best is None or (vm['macro_f1'],vm['balanced_accuracy'],vm['macro_auc'])>(best[0],best[1],best[2]): best=(vm['macro_f1'],vm['balanced_accuracy'],vm['macro_auc'],row,m,pv,pt)
df=pd.DataFrame(rows);df.to_csv(ROOT/'synthetic_quality_gate_grid.csv',index=False)
row=best[3]; print('BEST',row)
json.dump(row,open(ROOT/'optimized_synthetic_gate.json','w'),indent=2)
np.savez(ROOT/'optimized_gate_predictions.npz',y=yt,probs=best[6])
print(df.sort_values(['val_macro_f1','val_balanced_accuracy'],ascending=False).head(15).to_string(index=False))
