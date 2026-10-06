import numpy as np,pandas as pd,json
from sklearn.metrics import log_loss
from scipy.stats import beta
roi=np.load('/mnt/data/eswa_final_work/roi_results_fast/selected_dualview_predictions.npz',allow_pickle=True)
lat=np.load('/mnt/data/rev2work/inner/ESWA_GenAI_QA724_Reproducibility_Package/results/final_calibrated_predictions.npz',allow_pickle=True)
y=roi['y'].astype(int); pr=roi['probs']; pg=lat['probs']
assert np.array_equal(y,lat['y'])
K=4
ent=lambda p: -(p*np.log(np.clip(p,1e-12,1))).sum(1)/np.log(K)
yr=pr.argmax(1); yg=pg.argmax(1)
cr=pr.max(1); cg=pg.max(1); er=ent(pr); eg=ent(pg)
auto=(yr==yg)&(cr>=.70)&(cg>=.50)&(er<=.65)&(eg<=.75)
classes=['AOM','ASOM','CSOM','Normal']
rows=[]
for c,name in enumerate(classes):
  tc=(y==c); ret=tc&auto; ref=tc&~auto
  # precision on retained predictions (denominator retained predicted c); recall within retained true cases
  predc=(yr==c)&auto
  tp=((yr==c)&tc&auto).sum()
  prec=tp/predc.sum() if predc.sum() else None
  rec_within=tp/ret.sum() if ret.sum() else None
  overall_sens=tp/tc.sum() if tc.sum() else None
  errs=tc&(yr!=c); captured=(errs&~auto).sum(); totalerr=errs.sum()
  rows.append(dict(Class=name,Support=int(tc.sum()),Retained=int(ret.sum()),Referred=int(ref.sum()),Retained_fraction=float(ret.sum()/tc.sum()),Retained_TP=int(tp),Retained_predicted_as_class=int(predc.sum()),Retained_precision=prec,Retained_recall_among_retained=rec_within,Overall_auto_sensitivity=overall_sens,Class_errors=int(totalerr),Errors_captured_by_referral=int(captured),Error_capture_fraction=(captured/totalerr if totalerr else None)))
pd.DataFrame(rows).to_csv('/mnt/data/rev2work/results/referral_classwise.csv',index=False)
# overall
pred=yr; corr=pred==y; errors=~corr
print('auto',auto.sum(),'ref',(~auto).sum(),'acc',corr[auto].mean(),'error cap',(errors&~auto).sum()/errors.sum())
print(pd.DataFrame(rows))
# ECE 10 equal-width
def ece10(y,p):
 conf=p.max(1); pred=p.argmax(1); corr=(pred==y).astype(float)
 edges=np.linspace(0,1,11); e=0
 for i in range(10):
   m=(conf>=edges[i]) & ((conf<edges[i+1]) if i<9 else (conf<=edges[i+1]))
   if m.any(): e += m.mean()*abs(corr[m].mean()-conf[m].mean())
 return e

def brier_mc(y,p):
 oh=np.eye(p.shape[1])[y]
 return np.mean(np.sum((p-oh)**2,axis=1))
obs={'ece':ece10(y,pr),'brier':brier_mc(y,pr),'logloss':log_loss(y,pr,labels=np.arange(4))}
# image-level bootstrap 5000 resamples
rng=np.random.default_rng(20260921); vals=[]; refvals=[]; n=len(y)
for b in range(5000):
 idx=rng.integers(0,n,n); yy=y[idx]; pp=pr[idx]
 vals.append([ece10(yy,pp), brier_mc(yy,pp), log_loss(yy,pp,labels=np.arange(4))])
 # referral resample fixed decisions
 aa=auto[idx]; cc=corr[idx]; ee=errors[idx]
 coverage=aa.mean()
 racc=cc[aa].mean() if aa.any() else np.nan
 ecap=(ee&~aa).sum()/ee.sum() if ee.sum() else np.nan
 refvals.append([coverage,racc,ecap])
vals=np.array(vals); refvals=np.array(refvals,float)
res={'calibration_observed':obs,'calibration_bootstrap_95':{},'referral_bootstrap_95':{}}
for j,k in enumerate(['ece','brier','logloss']):res['calibration_bootstrap_95'][k]=[float(np.nanpercentile(vals[:,j],2.5)),float(np.nanpercentile(vals[:,j],97.5))]
for j,k in enumerate(['coverage','retained_accuracy','error_capture']):res['referral_bootstrap_95'][k]=[float(np.nanpercentile(refvals[:,j],2.5)),float(np.nanpercentile(refvals[:,j],97.5))]
json.dump(res,open('/mnt/data/rev2work/results/referral_calibration_uncertainty.json','w'),indent=2)
print(json.dumps(res,indent=2))
