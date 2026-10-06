from __future__ import annotations
import os, json, random, math, warnings
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd
from PIL import Image, ImageOps
import cv2, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import *
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(8)
ROOT=Path(os.environ.get('ESWA_QA724_ROOT', Path(__file__).resolve().parents[1])); DATA=ROOT/'Dataset/Dataset'; TRAIN_ROOT=DATA/'Training dataset/Otitis Media'; TEST_ROOT=DATA/'Testing dataset'; OUT=ROOT/'experiment_outputs'; OUT.mkdir(exist_ok=True)
CLASSES=['AOM','ASOM','CSOM','Normal']; C2I={c:i for i,c in enumerate(CLASSES)}
F2C={'AOM':'AOM','ASOM':'ASOM','CSOM':'CSOM','Normal Tympanic Membrane':'Normal'}
REMOVE={'AOM/aom (11).png','Normal Tympanic Membrane/normal (53).png','AOM/aom (8).png','Normal Tympanic Membrane/normal (51).png','AOM/aom (9).png','Normal Tympanic Membrane/normal (52).png','ASOM/asom (23).JPG','ASOM/asom (28).JPG','ASOM/asom (30).JPG','ASOM/asom (41).JPG','ASOM/asom (75).JPG','ASOM/asom (7).JPG'}

def crop(path,size=64):
    a=np.fromfile(str(path),np.uint8); b=cv2.imdecode(a,cv2.IMREAD_COLOR)
    if b is None: raise ValueError(path)
    g=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY); rgb=cv2.cvtColor(b,cv2.COLOR_BGR2RGB)
    m=(g>12).astype(np.uint8)*255; n,l,s,c=cv2.connectedComponentsWithStats(m,8)
    if n>1:
        h,w=g.shape; scores=[]
        for i in range(1,n):
            x,y,ww,hh,area=s[i]; cx,cy=c[i]; pen=((cx-w/2)**2+(cy-h/2)**2)/(h*h+w*w); scores.append((area*(1-.6*pen),x,y,ww,hh))
        _,x,y,ww,hh=max(scores); pad=int(.03*max(ww,hh)); x=max(0,x-pad);y=max(0,y-pad);x2=min(w,x+ww+2*pad);y2=min(h,y+hh+2*pad);rgb=rgb[y:y2,x:x2]
    return np.asarray(ImageOps.fit(Image.fromarray(rgb),(size,size),Image.Resampling.LANCZOS),np.uint8)

expert=[]
for f,c in F2C.items():
  for p in sorted((TRAIN_ROOT/f).glob('*')):
    rel=f'{f}/{p.name}'
    if p.is_file() and rel not in REMOVE: expert.append({'path':p,'label':c,'source':'expert','rel':rel})
assert len(expert)==687
idx=np.arange(len(expert)); y=np.array([C2I[r['label']] for r in expert])
tr,tmp=train_test_split(idx,test_size=.30,stratify=y,random_state=SEED); va,te=train_test_split(tmp,test_size=.5,stratify=y[tmp],random_state=SEED)
trr=[expert[i] for i in tr]; var=[expert[i] for i in va]; ter=[expert[i] for i in te]
pdf=pd.read_csv(ROOT/'test_ai_pseudo_labels_all_204.csv'); pseudo=[]
for _,r in pdf[pdf.included_in_QA_clean_724==True].iterrows(): pseudo.append({'path':TEST_ROOT/r.original_filename,'label':r.ai_predicted_label,'source':'pseudo'})
assert len(pseudo)==37
print('Split',len(trr),len(var),len(ter),Counter(r['label'] for r in trr),Counter(r['label'] for r in pseudo),flush=True)

# Same cVAE architecture as the completed generative run.
class CVAE(nn.Module):
  def __init__(self,latent=48,nc=4):
    super().__init__(); self.latent=latent; self.nc=nc
    self.enc=nn.Sequential(nn.Conv2d(3+nc,24,4,2,1),nn.BatchNorm2d(24),nn.LeakyReLU(.2),nn.Conv2d(24,48,4,2,1),nn.BatchNorm2d(48),nn.LeakyReLU(.2),nn.Conv2d(48,96,4,2,1),nn.BatchNorm2d(96),nn.LeakyReLU(.2),nn.Conv2d(96,128,4,2,1),nn.BatchNorm2d(128),nn.LeakyReLU(.2))
    self.fc_mu=nn.Linear(128*4*4,latent); self.fc_lv=nn.Linear(128*4*4,latent); self.fc_dec=nn.Linear(latent+nc,128*4*4)
    self.dec=nn.Sequential(nn.ConvTranspose2d(128,96,4,2,1),nn.BatchNorm2d(96),nn.ReLU(),nn.ConvTranspose2d(96,48,4,2,1),nn.BatchNorm2d(48),nn.ReLU(),nn.ConvTranspose2d(48,24,4,2,1),nn.BatchNorm2d(24),nn.ReLU(),nn.ConvTranspose2d(24,3,4,2,1),nn.Sigmoid())
  def encode(self,x,y):
    oh=F.one_hot(y,self.nc).float()[:,:,None,None].expand(-1,-1,64,64); h=self.enc(torch.cat([x,oh],1)).flatten(1); return self.fc_mu(h),self.fc_lv(h)
  def decode(self,z,y):
    oh=F.one_hot(y,self.nc).float(); return self.dec(self.fc_dec(torch.cat([z,oh],1)).view(-1,128,4,4))
vae=CVAE(); vae.load_state_dict(torch.load(OUT/'cvae_state.pt',map_location='cpu')); vae.eval()

def tensors(records):
  arr=np.stack([crop(r['path']) for r in records]); x=torch.from_numpy(arr.transpose(0,3,1,2)).float()/255.; y=np.array([C2I[r['label']] for r in records]); return x,y,arr

def hypothesis_features(x,batch=64):
  fs=[]
  with torch.no_grad():
    for st in range(0,len(x),batch):
      xb=x[st:st+batch]; parts=[]
      for c in range(4):
        yc=torch.full((len(xb),),c,dtype=torch.long); mu,_=vae.encode(xb,yc); parts.append(mu)
      f=torch.cat(parts,1); fs.append(f.numpy())
  return np.vstack(fs)

all_records=trr+var+ter+pseudo; Ximg,Yall,Arr=tensors(all_records); F_all=hypothesis_features(Ximg)
a=len(trr);b=a+len(var);c=b+len(ter)
Xtr,Xv,Xt,Xp=F_all[:a],F_all[a:b],F_all[b:c],F_all[c:]
ytr,yv,yt,yp=Yall[:a],Yall[a:b],Yall[b:c],Yall[c:]
np.savez_compressed(OUT/'cvae_hypothesis_embeddings.npz',Xtr=Xtr,Xv=Xv,Xt=Xt,Xp=Xp,ytr=ytr,yv=yv,yt=yt,yp=yp)

# Generate cVAE samples to reach 120 samples/class in QA train, then re-encode synthetic images without true-label leakage.
# Estimate class latent distribution from the matching conditional branch of expert training images.
with torch.no_grad():
  latent_by={i:[] for i in range(4)}
  xtr_img=Ximg[:a]
  for i in range(4):
    sel=np.where(ytr==i)[0]
    mu,_=vae.encode(xtr_img[sel],torch.full((len(sel),),i,dtype=torch.long)); latent_by[i]=mu.numpy()
base_count=Counter(list(ytr)+list(yp)); synth_imgs=[]; synth_y=[]
for i in [0,1,2]:
  n=max(0,120-base_count[i]); z0=latent_by[i]; mean=z0.mean(0); sd=z0.std(0)+1e-3
  # prototype-constrained truncated Gaussian sampling
  z=np.random.normal(mean,.55*sd,(n,z0.shape[1])).astype(np.float32); z=np.clip(z,mean-1.5*sd,mean+1.5*sd)
  with torch.no_grad(): imgs=vae.decode(torch.from_numpy(z),torch.full((n,),i,dtype=torch.long)).numpy()
  synth_imgs.append(torch.from_numpy(imgs)); synth_y.extend([i]*n)
Xsimg=torch.cat(synth_imgs,0); Xs=hypothesis_features(Xsimg); ys=np.array(synth_y)
np.savez_compressed(OUT/'cvae_synthetic_embeddings.npz',Xs=Xs,ys=ys)
print('Synthetic',Counter(ys),len(ys),flush=True)

# Utility metrics
def met(y,p):
  pr=p.argmax(1); pre,rec,f1,_=precision_recall_fscore_support(y,pr,average='macro',zero_division=0)
  auc=roc_auc_score(label_binarize(y,classes=np.arange(4)),p,multi_class='ovr',average='macro')
  return {'accuracy':accuracy_score(y,pr),'balanced_accuracy':balanced_accuracy_score(y,pr),'macro_precision':pre,'macro_recall':rec,'macro_f1':f1,'macro_auc':auc,'MCC':matthews_corrcoef(y,pr),'log_loss':log_loss(y,p,labels=np.arange(4))}

def fit_eval(name,model,Xtrain,ytrain):
  model.fit(Xtrain,ytrain); pv=model.predict_proba(Xv); pt=model.predict_proba(Xt); return {'name':name,'model':model,'val':met(yv,pv),'test':met(yt,pt),'pval':pv,'ptest':pt}

# Same-split architecture comparison; modest grid selected by validation macro-F1.
configs=[]
for C in [.1,1,10]: configs.append(('Logistic regression',Pipeline([('s',StandardScaler()),('m',LogisticRegression(C=C,max_iter=2500,class_weight='balanced',random_state=SEED))])))
for C,g in [(1,'scale'),(10,'scale'),(10,.001)]: configs.append(('RBF-SVM',Pipeline([('s',StandardScaler()),('m',SVC(C=C,gamma=g,probability=True,class_weight='balanced',random_state=SEED))])))
for n,mf in [(300,'sqrt'),(500,.7)]: configs.append(('Extra Trees',ExtraTreesClassifier(n_estimators=n,max_features=mf,class_weight='balanced',min_samples_leaf=2,random_state=SEED,n_jobs=-1)))
for n,mf in [(300,'sqrt'),(500,.7)]: configs.append(('Random Forest',RandomForestClassifier(n_estimators=n,max_features=mf,class_weight='balanced_subsample',min_samples_leaf=2,random_state=SEED,n_jobs=-1)))
for hid,alpha in [((64,),1e-3),((96,48),1e-3),((128,64),3e-4)]: configs.append(('Deep latent MLP',Pipeline([('s',StandardScaler()),('m',MLPClassifier(hidden_layer_sizes=hid,alpha=alpha,early_stopping=True,validation_fraction=.15,max_iter=600,random_state=SEED))])))

regimes={'Expert-only':(Xtr,ytr),'QA-724':(np.vstack([Xtr,Xp]),np.r_[ytr,yp]),'GenAI-augmented':(np.vstack([Xtr,Xp,Xs]),np.r_[ytr,yp,ys])}
allres=[]; best_by={}
for reg,(XX,yy) in regimes.items():
  cand=[]
  for family,model in configs:
    r=fit_eval(family,model,XX,yy); cand.append(r)
  # best config within family by val macro F1
  fams={}
  for r in cand:
    f=r['name']
    if f not in fams or r['val']['macro_f1']>fams[f]['val']['macro_f1']: fams[f]=r
  for f,r in fams.items(): allres.append({'regime':reg,'model':f,**{'val_'+k:v for k,v in r['val'].items()},**{'test_'+k:v for k,v in r['test'].items()}})
  best=max(fams.values(),key=lambda r:(r['val']['macro_f1'],r['val']['balanced_accuracy']))
  best_by[reg]=best
  print(reg,'best',best['name'],best['val'],best['test'],flush=True)
res=pd.DataFrame(allres); res.to_csv(OUT/'same_split_model_comparison.csv',index=False)

# Weighted probability ensemble for GenAI regime, weights proportional to validation macro-F1 among distinct families.
# Refit selected best family models and combine.
genXX,genyy=regimes['GenAI-augmented']; candidates=[]
for family in ['Logistic regression','RBF-SVM','Extra Trees','Deep latent MLP']:
  rows=[]
  for fam,model in configs:
    if fam==family: rows.append(fit_eval(fam,model,genXX,genyy))
  candidates.append(max(rows,key=lambda r:r['val']['macro_f1']))
w=np.array([max(0.01,r['val']['macro_f1']) for r in candidates]); w=w/w.sum()
pv=sum(wi*r['pval'] for wi,r in zip(w,candidates)); pt=sum(wi*r['ptest'] for wi,r in zip(w,candidates))
ensemble={'name':'Weighted probability ensemble','val':met(yv,pv),'test':met(yt,pt),'pval':pv,'ptest':pt,'weights':dict(zip([r['name'] for r in candidates],w.tolist()))}
print('Ensemble',ensemble['weights'],ensemble['val'],ensemble['test'],flush=True)

# Temperature scaling on ensemble probabilities using validation NLL.
def temp_scale(p,T):
  z=np.log(np.clip(p,1e-9,1))/T; z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)
opt=minimize_scalar(lambda T:log_loss(yv,temp_scale(pv,T),labels=np.arange(4)),bounds=(.25,4),method='bounded')
T=float(opt.x); pcal=temp_scale(pt,T); finalm=met(yt,pcal)
print('T',T,'final',finalm,flush=True)

# Save final predictions, reports, matrices, AUC.
np.savez(OUT/'final_calibrated_predictions.npz',y=yt,probs=pcal,temperature=T)
pred=pcal.argmax(1); cm=confusion_matrix(yt,pred,labels=np.arange(4)); pd.DataFrame(cm,index=CLASSES,columns=CLASSES).to_csv(OUT/'final_confusion_matrix.csv')
pd.DataFrame(classification_report(yt,pred,target_names=CLASSES,output_dict=True,zero_division=0)).T.to_csv(OUT/'final_classification_report.csv')
aucs={c:roc_auc_score((yt==i).astype(int),pcal[:,i]) for i,c in enumerate(CLASSES)}; json.dump(aucs,open(OUT/'final_class_auc.json','w'),indent=2)
json.dump({'temperature':T,'metrics':finalm,'ensemble_weights':ensemble['weights'],'validation_metrics':ensemble['val'],'uncalibrated_test_metrics':ensemble['test']},open(OUT/'final_model_summary.json','w'),indent=2)

# Selective referral analysis
ent=-(pcal*np.log(np.clip(pcal,1e-9,1))).sum(1)/math.log(4); conf=pcal.max(1); correct=(pred==yt)
rr=[]
for th in np.linspace(.30,.90,13):
 auto=(conf>=th)&(ent<=.65); rr.append({'confidence_threshold':th,'coverage':auto.mean(),'referral_rate':1-auto.mean(),'retained_accuracy':correct[auto].mean() if auto.any() else np.nan,'retained_n':int(auto.sum()),'error_capture_rate':((~correct)&(~auto)).sum()/max(1,(~correct).sum())})
pd.DataFrame(rr).to_csv(OUT/'selective_referral_analysis.csv',index=False)

# Bootstrap uncertainty
rng=np.random.default_rng(42); boots=[]; n=len(yt)
for b in range(2000):
 ix=rng.integers(0,n,n); yy=yt[ix]; pp=pcal[ix]
 if len(np.unique(yy))<4: continue
 boots.append({'replicate':b,**met(yy,pp)})
bdf=pd.DataFrame(boots); bdf.to_csv(OUT/'bootstrap_metrics_2000.csv',index=False)

# Generate manuscript figures
plt.figure(figsize=(7,4)); qcounts=pd.Series({'AOM':96,'ASOM':84,'CSOM':50,'Normal':494}); qcounts.plot(kind='bar'); plt.ylabel('Images'); plt.title('QA-clean 724 dataset class distribution'); plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(OUT/'fig_class_distribution.png',dpi=220); plt.close()
# comparison plot
pivot=res[res.regime=='GenAI-augmented'].sort_values('test_macro_f1'); plt.figure(figsize=(8,4.5)); x=np.arange(len(pivot)); plt.bar(x-.18,pivot.test_macro_f1,.36,label='Macro F1'); plt.bar(x+.18,pivot.test_balanced_accuracy,.36,label='Balanced accuracy'); plt.xticks(x,pivot.model,rotation=20,ha='right'); plt.ylim(0,1); plt.legend(); plt.title('GenAI-regime model comparison on expert-only test set'); plt.tight_layout(); plt.savefig(OUT/'fig_model_comparison.png',dpi=220); plt.close()
# confusion matrix
plt.figure(figsize=(5.2,4.4)); plt.imshow(cm,cmap='Blues'); plt.xticks(range(4),CLASSES);plt.yticks(range(4),CLASSES);plt.xlabel('Predicted');plt.ylabel('True');plt.title('Final calibrated ensemble confusion matrix');
for i in range(4):
 for j in range(4): plt.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
plt.colorbar();plt.tight_layout();plt.savefig(OUT/'fig_confusion_matrix.png',dpi=220);plt.close()
# ROC
from sklearn.metrics import roc_curve
plt.figure(figsize=(6,5));
for i,cname in enumerate(CLASSES):
 fpr,tpr,_=roc_curve((yt==i).astype(int),pcal[:,i]); plt.plot(fpr,tpr,label=f'{cname} (AUC={aucs[cname]:.3f})')
plt.plot([0,1],[0,1],'--',linewidth=1);plt.xlabel('False-positive rate');plt.ylabel('True-positive rate');plt.title('One-vs-rest ROC curves');plt.legend();plt.tight_layout();plt.savefig(OUT/'fig_roc_curves.png',dpi=220);plt.close()
# calibration reliability
from sklearn.calibration import calibration_curve
plt.figure(figsize=(6,5));
for i,cname in enumerate(CLASSES):
 frac,mean=calibration_curve((yt==i).astype(int),pcal[:,i],n_bins=7,strategy='quantile');plt.plot(mean,frac,marker='o',label=cname)
plt.plot([0,1],[0,1],'--');plt.xlabel('Mean predicted probability');plt.ylabel('Observed frequency');plt.title('Class-wise reliability diagram');plt.legend();plt.tight_layout();plt.savefig(OUT/'fig_calibration.png',dpi=220);plt.close()
# selective risk
sd=pd.read_csv(OUT/'selective_referral_analysis.csv');plt.figure(figsize=(6.5,4.5));plt.plot(sd.referral_rate,sd.retained_accuracy,marker='o');plt.xlabel('Referral rate');plt.ylabel('Accuracy among automated cases');plt.ylim(0,1.02);plt.title('Selective referral performance');plt.tight_layout();plt.savefig(OUT/'fig_selective_referral.png',dpi=220);plt.close()
# box and violin bootstrap
metrics_show=['accuracy','balanced_accuracy','macro_f1','macro_auc','MCC']; vals=[bdf[m].dropna().values for m in metrics_show]
plt.figure(figsize=(8,4.6));plt.boxplot(vals,tick_labels=['Accuracy','Balanced acc.','Macro F1','Macro AUC','MCC'],showfliers=False);plt.ylim(0,1.02);plt.title('Bootstrap distribution of final test metrics');plt.tight_layout();plt.savefig(OUT/'fig_boxplot_bootstrap.png',dpi=220);plt.close()
plt.figure(figsize=(8,4.6));plt.violinplot(vals,showmedians=True,showextrema=True);plt.xticks(range(1,6),['Accuracy','Balanced acc.','Macro F1','Macro AUC','MCC']);plt.ylim(0,1.02);plt.title('Violin plot of bootstrap test metrics');plt.tight_layout();plt.savefig(OUT/'fig_violin_bootstrap.png',dpi=220);plt.close()
# VAE history
vh=pd.read_csv(OUT/'cvae_training_history.csv');plt.figure(figsize=(6.5,4.2));plt.plot(vh.epoch,vh.reconstruction,label='Reconstruction');plt.plot(vh.epoch,vh.loss,label='Total loss');plt.xlabel('Epoch');plt.ylabel('Loss');plt.title('Conditional VAE training convergence');plt.legend();plt.tight_layout();plt.savefig(OUT/'fig_cvae_training.png',dpi=220);plt.close()

# summary table including ensemble
res2=pd.concat([res,pd.DataFrame([{'regime':'GenAI-augmented','model':'Weighted ensemble (calibrated)',**{'val_'+k:v for k,v in ensemble['val'].items()},**{'test_'+k:v for k,v in finalm.items()}}])],ignore_index=True);res2.to_csv(OUT/'complete_model_comparison.csv',index=False)
summary={'qa_total':724,'expert_clean':687,'pseudo':37,'train_expert':len(trr),'validation_expert':len(var),'test_expert':len(ter),'synthetic':len(ys),'class_counts_qa724':dict(Counter([r['label'] for r in expert+pseudo])),'final_metrics':finalm,'class_auc':aucs,'temperature':T,'weights':ensemble['weights'],'removed':sorted(REMOVE)}
json.dump(summary,open(OUT/'experiment_summary.json','w'),indent=2)
print('DONE',OUT,flush=True)
