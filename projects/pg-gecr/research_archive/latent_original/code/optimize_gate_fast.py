import os
import numpy as np,pandas as pd,json
from pathlib import Path
from sklearn.preprocessing import StandardScaler,label_binarize
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import *
R=Path(os.environ.get('ESWA_QA724_ROOT', Path(__file__).resolve().parents[1]))/'experiment_outputs';a=np.load(R/'cvae_hypothesis_embeddings.npz');s=np.load(R/'cvae_synthetic_embeddings.npz')
Xtr,Xv,Xt,Xp=[a[k] for k in ['Xtr','Xv','Xt','Xp']];ytr,yv,yt,yp=[a[k] for k in ['ytr','yv','yt','yp']];Xs,ys=s['Xs'],s['ys']
def met(y,p):
 q=p.argmax(1);pre,rec,f1,_=precision_recall_fscore_support(y,q,average='macro',zero_division=0)
 return {'accuracy':accuracy_score(y,q),'balanced_accuracy':balanced_accuracy_score(y,q),'macro_precision':pre,'macro_recall':rec,'macro_f1':f1,'macro_auc':roc_auc_score(label_binarize(y,classes=np.arange(4)),p,multi_class='ovr',average='macro'),'MCC':matthews_corrcoef(y,q),'log_loss':log_loss(y,p,labels=np.arange(4))}
rank={}
for c in [0,1,2]:
 ex=Xtr[ytr==c];mu=ex.mean(0);sd=ex.std(0)+1e-3;ids=np.where(ys==c)[0];d=np.mean(((Xs[ids]-mu)/sd)**2,1);rank[c]=ids[np.argsort(d)]
baseX=np.vstack([Xtr,Xp]);basey=np.r_[ytr,yp]
configs=[(0,0,0),(.1,.1,.1),(.25,.25,.25),(.5,.5,.5),(.75,.75,.75),(1,1,1),(0,0,.25),(0,0,.5),(0,0,1),(.25,0,.5),(0,.25,.5),(.25,.25,.5)]
models={'Logistic':lambda:Pipeline([('s',StandardScaler()),('m',LogisticRegression(C=.1,max_iter=2000,class_weight='balanced',random_state=42))]),'ExtraTrees':lambda:ExtraTreesClassifier(n_estimators=250,max_features='sqrt',class_weight='balanced',min_samples_leaf=2,random_state=42,n_jobs=-1)}
rows=[];best=None
for fs in configs:
 ids=np.concatenate([rank[c][:round(len(rank[c])*fs[c])] for c in [0,1,2]]) if any(fs) else np.array([],int)
 XX=np.vstack([baseX,Xs[ids]]) if len(ids) else baseX;yy=np.r_[basey,ys[ids]] if len(ids) else basey
 for name,fn in models.items():
  m=fn();m.fit(XX,yy);pv=m.predict_proba(Xv);pt=m.predict_proba(Xt);vm=met(yv,pv);tm=met(yt,pt);row={'model':name,'f_AOM':fs[0],'f_ASOM':fs[1],'f_CSOM':fs[2],'n_synth':len(ids),**{'val_'+k:v for k,v in vm.items()},**{'test_'+k:v for k,v in tm.items()}};rows.append(row)
  key=(vm['macro_f1'],vm['balanced_accuracy'],vm['macro_auc'])
  if best is None or key>best[0]:best=(key,row,pt)
df=pd.DataFrame(rows);df.to_csv(R/'synthetic_quality_gate_grid.csv',index=False);json.dump(best[1],open(R/'optimized_synthetic_gate.json','w'),indent=2);np.savez(R/'optimized_gate_predictions.npz',y=yt,probs=best[2]);print('BEST',best[1]);print(df.sort_values('val_macro_f1',ascending=False).head(10).to_string(index=False))
