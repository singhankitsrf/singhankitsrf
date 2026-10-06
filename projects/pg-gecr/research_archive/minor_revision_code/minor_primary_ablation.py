import numpy as np, pandas as pd, json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from scipy.optimize import minimize_scalar
OUT=Path('/mnt/data/minor_work/minor_results');OUT.mkdir(exist_ok=True)
Z=np.load('/mnt/data/eswa_final_work/roi_results_fast/features.npz');X=Z['X'];Y=Z['Y'];a,b,c=480,583,687
Xtr,Xv,Xt=X[:a],X[a:b],X[b:c]; ytr,yv,yt=Y[:a],Y[a:b],Y[b:c]
# Expert-only ROI model reproduced from archived selected pipeline
m=Pipeline([('s',StandardScaler()),('p',PCA(n_components=.97,whiten=True,random_state=42)),('m',LogisticRegression(C=.05,max_iter=4000,class_weight='balanced',random_state=42))])
m.fit(Xtr,ytr); pval=m.predict_proba(Xv); ptest=m.predict_proba(Xt)
def ts(p,T):
 z=np.log(np.clip(p,1e-9,1))/T; z-=z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)
T=float(minimize_scalar(lambda T:log_loss(yv,ts(pval,T),labels=np.arange(4)),bounds=(.25,4),method='bounded').x)
pr=ts(ptest,T)
pg=np.load('/mnt/data/minor_work/orig/ESWA_GenAI_QA724_Reproducibility_Package/results/final_calibrated_predictions.npz')['probs']
yg=np.load('/mnt/data/minor_work/orig/ESWA_GenAI_QA724_Reproducibility_Package/results/final_calibrated_predictions.npz')['y']
assert np.array_equal(yt,yg)
CL=['AOM','ASOM','CSOM','Normal']
rpred=pr.argmax(1); gpred=pg.argmax(1); correct=rpred==yt
ent_r=-(pr*np.log(np.clip(pr,1e-9,1))).sum(1)/np.log(4); ent_g=-(pg*np.log(np.clip(pg,1e-9,1))).sum(1)/np.log(4)
# Full expert-ROI + latent consensus gate (same guardrails)
full=(rpred==gpred)&(pr.max(1)>=.70)&(pg.max(1)>=.50)&(ent_r<=.65)&(ent_g<=.75)
# ROI-only fixed rule removes latent constraints
roi_fixed=(pr.max(1)>=.70)&(ent_r<=.65)
# Post-hoc matched-coverage ROI-only confidence-ranking ablation: retain exactly full N highest confidence.
n=int(full.sum()); order=np.argsort(-pr.max(1)); roi_matched=np.zeros(len(yt),bool); roi_matched[order[:n]]=True

def summary(mask,name):
    retained=int(mask.sum()); cov=mask.mean(); acc=float(correct[mask].mean()) if retained else np.nan
    errs=(~correct); ec=float((errs & ~mask).sum()/errs.sum())
    rows=[]
    for ci,cn in enumerate(CL):
      truth=yt==ci; rt=truth & mask; ntrue=int(truth.sum()); nr=int(rt.sum()); nref=ntrue-nr
      tp=int((rt)&(rpred==ci).sum()) if False else int((rt & (rpred==ci)).sum())
      sens=float(tp/ntrue) if ntrue else np.nan # retained-correct / all true class, clinically interpretable auto-correct sensitivity
      racc=float((rpred[rt]==ci).mean()) if nr else np.nan
      pred_ret=(rpred==ci)&mask; prec=float((yt[pred_ret]==ci).mean()) if pred_ret.any() else np.nan
      rows.append(dict(rule=name,Class=cn,support=ntrue,retained=nr,referred=nref,retained_fraction=nr/ntrue if ntrue else np.nan,auto_correct_sensitivity=sens,retained_case_accuracy=racc,retained_precision=prec))
    return dict(rule=name,coverage=float(cov),retained_n=retained,retained_accuracy=acc,error_capture=float(ec),errors_total=int(errs.sum()),errors_referred=int((errs&~mask).sum())),rows
outs=[];classrows=[]
for mask,name in [(roi_fixed,'Expert-ROI only, fixed confidence/entropy'),(full,'Expert-ROI + latent consensus'),(roi_matched,'Expert-ROI only, post-hoc matched coverage')]:
    s,rr=summary(mask,name); outs.append(s); classrows+=rr
# coverage-risk curves based on expert ROI confidence and full consensus varying ROI threshold
curve=[]
for th in [0.50,0.60,0.70,0.75,0.80,0.85,0.90]:
  ro=(pr.max(1)>=th)&(ent_r<=.65)
  fu=(rpred==gpred)&(pr.max(1)>=th)&(pg.max(1)>=.50)&(ent_r<=.65)&(ent_g<=.75)
  for mask,name in [(ro,'ROI-only'),(fu,'ROI+latent')]:
    curve.append({'rule':name,'roi_threshold':th,'coverage':float(mask.mean()),'retained_accuracy':float(correct[mask].mean()) if mask.any() else np.nan,'error_capture':float(((~correct)&(~mask)).sum()/max(1,(~correct).sum())),'retained_n':int(mask.sum())})
# bootstrap paired at matched full coverage by resampling indices and applying precomputed masks; this is descriptive, not patient-clustered
rng=np.random.default_rng(20260927); boot=[]
for b0 in range(10000):
 ix=rng.integers(0,len(yt),len(yt))
 for mask,name in [(roi_matched,'ROI-only matched'),(full,'ROI+latent')]:
   mm=mask[ix]; cc=correct[ix]; er=~cc
   boot.append((b0,name,mm.mean(),cc[mm].mean() if mm.any() else np.nan,((er)&(~mm)).sum()/max(1,er.sum())))
bdf=pd.DataFrame(boot,columns=['rep','rule','coverage','retained_accuracy','error_capture'])
ci={}
for name,g in bdf.groupby('rule'):
 ci[name]={k:[float(g[k].quantile(.025)),float(g[k].quantile(.975))] for k in ['coverage','retained_accuracy','error_capture']}
# paired differences full - matched ROI-only within same resample
piv=bdf.pivot(index='rep',columns='rule',values=['retained_accuracy','error_capture'])
diffs={}
for k in ['retained_accuracy','error_capture']:
 d=piv[k]['ROI+latent']-piv[k]['ROI-only matched'];diffs[k]={'point':outs[1][k]-outs[2][k],'ci95':[float(d.quantile(.025)),float(d.quantile(.975))]}
# expert metrics
from sklearn.preprocessing import label_binarize
metrics={'accuracy':float(accuracy_score(yt,rpred)),'balanced_accuracy':float(balanced_accuracy_score(yt,rpred)),'macro_f1':float(f1_score(yt,rpred,average='macro')),'macro_auc':float(roc_auc_score(label_binarize(yt,classes=np.arange(4)),pr,multi_class='ovr',average='macro')),'mcc':float(matthews_corrcoef(yt,rpred)),'log_loss':float(log_loss(yt,pr,labels=np.arange(4))),'temperature':T}
json.dump({'expert_metrics':metrics,'rules':outs,'bootstrap95':ci,'paired_diff_full_minus_roi_matched':diffs},open(OUT/'expert_roi_referral_ablation.json','w'),indent=2)
pd.DataFrame(classrows).to_csv(OUT/'expert_roi_referral_classwise.csv',index=False)
pd.DataFrame(curve).to_csv(OUT/'expert_roi_referral_curve.csv',index=False)
np.savez_compressed(OUT/'expert_roi_probabilities.npz',y=yt,probs=pr,yval=yv,pval=ts(pval,T),temperature=T,full_mask=full,roi_fixed_mask=roi_fixed,roi_matched_mask=roi_matched)
print(json.dumps({'expert_metrics':metrics,'rules':outs,'bootstrap95':ci,'paired_diff_full_minus_roi_matched':diffs},indent=2))
print(pd.DataFrame(classrows).to_string(index=False))
print(pd.DataFrame(curve).to_string(index=False))
