import json, math
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import *
from sklearn.preprocessing import label_binarize
from scipy.stats import binomtest
OUT=Path('/mnt/data/eswa_final_work/final_stats');OUT.mkdir(exist_ok=True)
roi=np.load('/mnt/data/eswa_final_work/roi_results_fast/selected_dualview_predictions.npz')
gen=np.load('/mnt/data/eswa_repro/ESWA_GenAI_QA724_Reproducibility_Package/results/final_calibrated_predictions.npz')
y=roi['y'];pr=roi['probs'];pg=gen['probs'];assert np.array_equal(y,gen['y'])
CL=['AOM','ASOM','CSOM','Normal']
def met(y,p):
 pred=p.argmax(1);pre,rec,f1,_=precision_recall_fscore_support(y,pred,average='macro',zero_division=0);auc=roc_auc_score(label_binarize(y,classes=np.arange(4)),p,multi_class='ovr',average='macro')
 return {'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred),'macro_precision':pre,'macro_recall':rec,'macro_f1':f1,'macro_auc':auc,'MCC':matthews_corrcoef(y,pred),'log_loss':log_loss(y,p,labels=np.arange(4))}
roi_m=met(y,pr);gen_m=met(y,pg)
rpred=pr.argmax(1);gpred=pg.argmax(1)
# classwise AUC/report
rep=classification_report(y,rpred,target_names=CL,output_dict=True,zero_division=0)
aucs={c:roc_auc_score((y==i).astype(int),pr[:,i]) for i,c in enumerate(CL)}
pd.DataFrame(confusion_matrix(y,rpred),index=CL,columns=CL).to_csv(OUT/'final_confusion_matrix.csv')
pd.DataFrame(rep).T.to_csv(OUT/'final_classification_report.csv')
json.dump(aucs,open(OUT/'final_class_auc.json','w'),indent=2)
# agreement/consensus referral table
ent_r=-(pr*np.log(np.clip(pr,1e-9,1))).sum(1)/np.log(4); ent_g=-(pg*np.log(np.clip(pg,1e-9,1))).sum(1)/np.log(4)
rows=[]
for th in [0.50,0.60,0.70,0.75,0.80,0.85,0.90]:
 auto=(rpred==gpred)&(pr.max(1)>=th)&(pg.max(1)>=0.50)&(ent_r<=0.65)&(ent_g<=0.75)
 correct=(rpred==y)
 rows.append({'roi_confidence_threshold':th,'coverage':auto.mean(),'referral_rate':1-auto.mean(),'retained_n':int(auto.sum()),'retained_accuracy':correct[auto].mean() if auto.any() else np.nan,'error_capture_rate':((~correct)&(~auto)).sum()/max(1,(~correct).sum())})
pd.DataFrame(rows).to_csv(OUT/'consensus_referral.csv',index=False)
# binary disease vs normal
truth=(y!=3).astype(int);pdisease=1-pr[:,3]
brows=[]
for th in [.3,.4,.5,.6,.7]:
 pred=(pdisease>=th).astype(int);tn,fp,fn,tp=confusion_matrix(truth,pred).ravel();brows.append({'threshold':th,'accuracy':accuracy_score(truth,pred),'sensitivity':tp/(tp+fn),'specificity':tn/(tn+fp),'precision':precision_score(truth,pred,zero_division=0),'f1':f1_score(truth,pred),'TP':tp,'FN':fn,'TN':tn,'FP':fp})
pd.DataFrame(brows).to_csv(OUT/'binary_screening.csv',index=False)
# Calibration
conf=pr.max(1);corr=rpred==y
bins=np.linspace(0,1,11);ece=0;cal=[]
for lo,hi in zip(bins[:-1],bins[1:]):
 mask=(conf>=lo)&(conf<hi if hi<1 else conf<=hi)
 if mask.any():
  acc=corr[mask].mean();c=conf[mask].mean();ece+=mask.mean()*abs(acc-c);cal.append({'lower':lo,'upper':hi,'n':int(mask.sum()),'accuracy':acc,'mean_confidence':c})
pd.DataFrame(cal).to_csv(OUT/'calibration_bins.csv',index=False)
one=np.eye(4)[y];brier=float(np.mean(np.sum((pr-one)**2,axis=1)))
# McNemar exact ROI vs GenAI
rc=(rpred==y);gc=(gpred==y);b=int((rc & ~gc).sum());c=int((~rc & gc).sum());pval=float(binomtest(min(b,c),n=b+c,p=.5,alternative='two-sided').pvalue) if b+c else 1
# bootstrap
rng=np.random.default_rng(2026);boots=[];n=len(y)
for k in range(2000):
 ix=rng.integers(0,n,n);yy=y[ix]
 if len(np.unique(yy))<4:continue
 m=met(yy,pr[ix]);
 # consensus at .70
 rp=rpred[ix];gp=gpred[ix];er=ent_r[ix];eg=ent_g[ix];pp=pr[ix];qq=pg[ix];auto=(rp==gp)&(pp.max(1)>=.70)&(qq.max(1)>=.50)&(er<=.65)&(eg<=.75);correct=(rp==yy)
 m.update({'consensus_coverage':auto.mean(),'consensus_retained_accuracy':correct[auto].mean() if auto.any() else np.nan,'consensus_error_capture':((~correct)&(~auto)).sum()/max(1,(~correct).sum()),'binary_auc':roc_auc_score((yy!=3).astype(int),1-pp[:,3])})
 boots.append(m)
bdf=pd.DataFrame(boots);bdf.to_csv(OUT/'bootstrap_2000.csv',index=False)
summary={}
for col in ['accuracy','balanced_accuracy','macro_f1','macro_auc','MCC','log_loss','consensus_coverage','consensus_retained_accuracy','consensus_error_capture','binary_auc']:
 v=bdf[col].dropna();summary[col]={'median':float(v.median()),'lower_95':float(v.quantile(.025)),'upper_95':float(v.quantile(.975))}
# performance difference bootstrap ROI - GenAI
D=[]
for k in range(2000):
 ix=rng.integers(0,n,n);yy=y[ix]
 if len(np.unique(yy))<4:continue
 a=met(yy,pr[ix]);g=met(yy,pg[ix]);D.append({q:a[q]-g[q] for q in ['accuracy','balanced_accuracy','macro_f1','macro_auc','MCC','log_loss']})
dfD=pd.DataFrame(D);dfD.to_csv(OUT/'bootstrap_difference_roi_minus_genai.csv',index=False)
diff={col:{'point':roi_m[col]-gen_m[col],'median':float(dfD[col].median()),'lower_95':float(dfD[col].quantile(.025)),'upper_95':float(dfD[col].quantile(.975))} for col in dfD.columns}
final={'roi_metrics':roi_m,'genai_metrics':gen_m,'delta_roi_minus_genai':{k:roi_m[k]-gen_m[k] for k in roi_m},'class_auc':aucs,'binary_auc':roc_auc_score(truth,pdisease),'binary_average_precision':average_precision_score(truth,pdisease),'binary_brier':brier_score_loss(truth,pdisease),'multiclass_ECE':ece,'multiclass_Brier':brier,'mcnemar':{'roi_correct_genai_wrong':b,'roi_wrong_genai_correct':c,'exact_p':pval},'bootstrap':summary,'bootstrap_differences':diff}
json.dump(final,open(OUT/'final_statistics.json','w'),indent=2)
print(json.dumps(final,indent=2))
