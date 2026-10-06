from pathlib import Path
import pandas as pd,numpy as np,sys
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score,accuracy_score,balanced_accuracy_score,roc_auc_score
from joblib import Parallel,delayed
OUT=Path('/mnt/data/minor_work/minor_results');z=np.load(OUT/'aux_original_split_arrays.npz');Xtr,ytr,Xv,yv,Xt,yt=[z[k] for k in ['Xtr','ytr','Xv','yv','Xt','yt']]
def features(X):return np.concatenate([X[:,:,::4,::4].reshape(len(X),-1),X.mean((2,3)),X.std((2,3))],1).astype(np.float32)
Ftr,Fv,Ft=features(Xtr),features(Xv),features(Xt);BASE=20260921;t=int(sys.argv[1]);reps=int(sys.argv[2]);start=int(sys.argv[3]) if len(sys.argv)>3 else 0
def one(repid):
 rg=np.random.default_rng(BASE+1000*t+repid);ids=[]
 for c in range(5):
  ix=np.where(ytr==c)[0].copy();rg.shuffle(ix);ids.extend(ix[:t].tolist())
 ids=np.array(ids);rg.shuffle(ids);m=ExtraTreesClassifier(n_estimators=300,max_features=.5,class_weight='balanced',n_jobs=1,random_state=BASE);m.fit(Ftr[ids],ytr[ids]);vp=m.predict(Fv);tp=m.predict(Ft);pro=m.predict_proba(Ft)
 return {'per_class_training':t,'repeat':repid,'subset_seed':BASE+1000*t+repid,'n_train':len(ids),'val_macro_f1':f1_score(yv,vp,average='macro',zero_division=0),'test_macro_f1':f1_score(yt,tp,average='macro',zero_division=0),'test_accuracy':accuracy_score(yt,tp),'test_balanced_accuracy':balanced_accuracy_score(yt,tp),'test_macro_auc':roc_auc_score(yt,pro,multi_class='ovr',average='macro')}
res=Parallel(n_jobs=min(reps,4),backend='threading')(delayed(one)(r) for r in range(start,start+reps));df=pd.DataFrame(res);df.to_csv(OUT/f'curve_t{t}.csv',index=False);print(df.to_string(index=False));print('SUMMARY',t,df.test_macro_f1.mean(),df.test_macro_f1.std(ddof=1),df.test_macro_f1.min(),df.test_macro_f1.max())
