import json,warnings
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler,label_binarize
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from scipy.optimize import minimize_scalar
warnings.filterwarnings('ignore')
SEED=42;OUT=Path('/mnt/data/eswa_final_work/roi_results_fast');z=np.load(OUT/'features.npz');X=z['X'];Y=z['Y'];a,b,c=480,583,687
Xtr,Xv,Xt,Xp=X[:a],X[a:b],X[b:c],X[c:];ytr,yv,yt,yp=Y[:a],Y[a:b],Y[b:c],Y[c:]
def met(y,p):
 pred=p.argmax(1); pre,rec,f1,_=precision_recall_fscore_support(y,pred,average='macro',zero_division=0);auc=roc_auc_score(label_binarize(y,classes=np.arange(4)),p,multi_class='ovr',average='macro')
 return {'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred),'macro_precision':pre,'macro_recall':rec,'macro_f1':f1,'macro_auc':auc,'MCC':matthews_corrcoef(y,pred),'log_loss':log_loss(y,p,labels=np.arange(4))}
def model(C):return Pipeline([('s',StandardScaler()),('p',PCA(n_components=.97,whiten=True,random_state=SEED)),('m',LogisticRegression(C=C,max_iter=4000,class_weight='balanced',random_state=SEED))])
models=[]
for name,C,XX,yy in [('Expert ROI',.05,Xtr,ytr),('QA ROI',.03,np.vstack([Xtr,Xp]),np.r_[ytr,yp]),('Expert ROI high-C',1,Xtr,ytr),('QA ROI mid-C',.5,np.vstack([Xtr,Xp]),np.r_[ytr,yp])]:
 m=model(C);m.fit(XX,yy);pv=m.predict_proba(Xv);pt=m.predict_proba(Xt);models.append((name,pv,pt,met(yv,pv),met(yt,pt)));print(name,models[-1][3],models[-1][4])
# grid of validation-derived weighted mixtures of expert and QA models, choose by val macro F1 then NLL
cands=[]
for i in range(len(models)):
 for j in range(i+1,len(models)):
  for w in np.linspace(0,1,21):
   pv=w*models[i][1]+(1-w)*models[j][1];pt=w*models[i][2]+(1-w)*models[j][2];mv=met(yv,pv);mt=met(yt,pt);cands.append((mv['macro_f1'],mv['balanced_accuracy'],-mv['log_loss'],i,j,w,pv,pt,mv,mt))
best=max(cands,key=lambda x:x[:3]);_,_,_,i,j,w,pv,pt,mv,mt=best
print('BEST MIX',models[i][0],models[j][0],w,mv,mt)
def ts(p,T):
 z=np.log(np.clip(p,1e-9,1))/T;z-=z.max(1,keepdims=True);e=np.exp(z);return e/e.sum(1,keepdims=True)
T=float(minimize_scalar(lambda T:log_loss(yv,ts(pv,T),labels=np.arange(4)),bounds=(.25,4),method='bounded').x);pc=ts(pt,T);final=met(yt,pc);print('CAL',T,final)
np.savez_compressed(OUT/'selected_dualview_predictions.npz',y=yt,probs=pc,yval=yv,pval=ts(pv,T),temperature=T)
json.dump({'members':[models[i][0],models[j][0]],'weight_first':float(w),'validation':mv,'uncalibrated_test':mt,'temperature':T,'test':final,'individual':{m[0]:{'validation':m[3],'test':m[4]} for m in models}},open(OUT/'selected_dualview_summary.json','w'),indent=2)
# reports
pred=pc.argmax(1);pd.DataFrame(confusion_matrix(yt,pred),index=['AOM','ASOM','CSOM','Normal'],columns=['AOM','ASOM','CSOM','Normal']).to_csv(OUT/'selected_confusion_matrix.csv');pd.DataFrame(classification_report(yt,pred,target_names=['AOM','ASOM','CSOM','Normal'],output_dict=True,zero_division=0)).T.to_csv(OUT/'selected_classification_report.csv')
