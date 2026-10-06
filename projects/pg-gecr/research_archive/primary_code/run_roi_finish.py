import json,warnings
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,label_binarize
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import *
from scipy.optimize import minimize_scalar
warnings.filterwarnings('ignore')
SEED=42;OUT=Path('/mnt/data/eswa_final_work/roi_results_fast');z=np.load(OUT/'features.npz');X=z['X'];Y=z['Y'];a,b,c=480,583,687
Xtr,Xv,Xt,Xp=X[:a],X[a:b],X[b:c],X[c:];ytr,yv,yt,yp=Y[:a],Y[a:b],Y[b:c],Y[c:]
def met(y,p):
 pred=p.argmax(1); pre,rec,f1,_=precision_recall_fscore_support(y,pred,average='macro',zero_division=0);auc=roc_auc_score(label_binarize(y,classes=np.arange(4)),p,multi_class='ovr',average='macro')
 return {'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred),'macro_precision':pre,'macro_recall':rec,'macro_f1':f1,'macro_auc':auc,'MCC':matthews_corrcoef(y,pred),'log_loss':log_loss(y,p,labels=np.arange(4))}
def fit(name,m,XX,yy):m.fit(XX,yy);pv=m.predict_proba(Xv);pt=m.predict_proba(Xt);return {'name':name,'pval':pv,'ptest':pt,'val':met(yv,pv),'test':met(yt,pt)}
configs=[]
for C in [.03,.05,.1,.2,.5,1,2,5]:configs.append((f'ROI Logistic C={C}',Pipeline([('s',StandardScaler()),('p',PCA(n_components=.97,whiten=True,random_state=SEED)),('m',LogisticRegression(C=C,max_iter=4000,class_weight='balanced',random_state=SEED))])))
for n,mf,leaf in [(500,'sqrt',1),(700,.5,1),(700,.7,2)]:configs.append((f'ROI ExtraTrees {n}-{mf}-{leaf}',ExtraTreesClassifier(n_estimators=n,max_features=mf,min_samples_leaf=leaf,class_weight='balanced',random_state=SEED,n_jobs=-1)))
try:
 from lightgbm import LGBMClassifier
 for leaves in [15,31]:configs.append((f'ROI LightGBM leaves={leaves}',LGBMClassifier(n_estimators=250,learning_rate=.03,num_leaves=leaves,max_depth=7,subsample=.85,colsample_bytree=.75,class_weight='balanced',reg_lambda=2,random_state=SEED,verbosity=-1,n_jobs=-1)))
except Exception: pass
rows=[];results={}
for reg,XX,yy in [('Expert-only',Xtr,ytr),('QA-724',np.vstack([Xtr,Xp]),np.r_[ytr,yp])]:
 rs=[]
 for name,m in configs:
  r=fit(name,m,XX,yy);rs.append(r);print(reg,name,round(r['val']['macro_f1'],3),round(r['test']['macro_f1'],3),flush=True)
 # top by validation
 best=max(rs,key=lambda r:(r['val']['macro_f1'],r['val']['balanced_accuracy']))
 results[reg]=rs
 for r in rs: rows.append({'regime':reg,'model':r['name'],**{'val_'+k:v for k,v in r['val'].items()},**{'test_'+k:v for k,v in r['test'].items()}})
 print('BEST',reg,best['name'],best['val'],best['test'],flush=True)
# validation-weighted combination of top 4 QA models, remove near duplicates by model family
qa=results['QA-724']; top=sorted(qa,key=lambda r:r['val']['macro_f1'],reverse=True)[:5]
w=np.array([r['val']['macro_f1']**4 for r in top]);w/=w.sum();pv=sum(wi*r['pval'] for wi,r in zip(w,top));pt=sum(wi*r['ptest'] for wi,r in zip(w,top));val=met(yv,pv);unc=met(yt,pt)
def ts(p,T):
 q=np.log(np.clip(p,1e-9,1))/T;q-=q.max(1,keepdims=True);e=np.exp(q);return e/e.sum(1,keepdims=True)
T=float(minimize_scalar(lambda T:log_loss(yv,ts(pv,T),labels=np.arange(4)),bounds=(.25,4),method='bounded').x);pct=ts(pt,T);final=met(yt,pct)
rows.append({'regime':'QA-724','model':'ROI validation-weighted fusion (calibrated)',**{'val_'+k:v for k,v in val.items()},**{'test_'+k:v for k,v in final.items()}})
pd.DataFrame(rows).to_csv(OUT/'model_comparison.csv',index=False);np.savez_compressed(OUT/'final_predictions.npz',y=yt,probs=pct,yval=yv,pval=pv,temperature=T)
json.dump({'weights':dict(zip([r['name'] for r in top],w.tolist())),'validation':val,'uncalibrated_test':unc,'test':final,'temperature':T},open(OUT/'summary.json','w'),indent=2)
print('FUSION',dict(zip([r['name'] for r in top],w)),val,final,flush=True)
