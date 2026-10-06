import os
from pathlib import Path
import json, math
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, matthews_corrcoef
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve
from PIL import Image, ImageOps
import cv2
P=Path(os.environ.get('ESWA_QA724_ROOT', Path(__file__).resolve().parents[1])); R=P/'experiment_outputs'
CL=['AOM','ASOM','CSOM','Normal']
# load predictions
z=np.load(R/'final_calibrated_predictions.npz'); y=z['y']; p=z['probs']; pred=p.argmax(1)
# Bootstrap
rng=np.random.default_rng(42); rows=[]; n=len(y)
for b in range(600):
    ix=rng.integers(0,n,n); yy=y[ix]; pp=p[ix]
    if len(np.unique(yy))<4: continue
    q=pp.argmax(1); pre,rec,f1,_=precision_recall_fscore_support(yy,q,average='macro',zero_division=0)
    auc=roc_auc_score(label_binarize(yy,classes=np.arange(4)),pp,multi_class='ovr',average='macro')
    rows.append({'replicate':b,'accuracy':accuracy_score(yy,q),'balanced_accuracy':balanced_accuracy_score(yy,q),'macro_f1':f1,'macro_auc':auc,'MCC':matthews_corrcoef(yy,q)})
b=pd.DataFrame(rows); b.to_csv(R/'bootstrap_metrics_600.csv',index=False)
# class distribution
counts=pd.Series({'AOM':96,'ASOM':84,'CSOM':50,'Normal':494})
plt.figure(figsize=(6.8,4.2)); counts.plot(kind='bar'); plt.ylabel('Number of images'); plt.title('QA-clean 724 dataset class distribution'); plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(R/'fig_class_distribution.png',dpi=220); plt.close()
# dataset flow
fig,ax=plt.subplots(figsize=(10,4.4)); ax.axis('off')
boxes=[(.03,.55,.18,.25,'Original expert-labelled\ncohort\n699 images'),(.28,.68,.18,.19,'Remove 6 cross-label\nconflict images'),(.28,.37,.18,.19,'Remove 6 redundant\nsame-label copies'),(.53,.55,.18,.25,'QA-clean expert core\n687 images'),(.53,.12,.18,.20,'Conservative pseudo-labels\n37 images\n(3 AOM, 2 CSOM, 32 Normal)'),(.79,.55,.18,.25,'QA-clean research set\n724 images')]
for x,y0,w,h,t in boxes:
    ax.add_patch(FancyBboxPatch((x,y0),w,h,boxstyle='round,pad=.012',linewidth=1.2,facecolor='white')); ax.text(x+w/2,y0+h/2,t,ha='center',va='center',fontsize=9)
for a1,b1,a2,b2 in [(.21,.675,.28,.775),(.21,.675,.28,.465),(.46,.775,.53,.675),(.46,.465,.53,.675),(.62,.32,.62,.55),(.71,.675,.79,.675)]:
    ax.add_patch(FancyArrowPatch((a1,b1),(a2,b2),arrowstyle='-|>',mutation_scale=13,linewidth=1.2))
ax.set_title('Dataset quality-assurance and provenance flow',fontsize=12); plt.tight_layout(); plt.savefig(R/'fig_dataset_flow.png',dpi=220,bbox_inches='tight'); plt.close()
# architecture diagram
fig,ax=plt.subplots(figsize=(12,6)); ax.axis('off')
items=[(.02,.57,.15,.22,'Input and QA gate\nField crop, resize, duplicate audit, provenance'),(.21,.57,.16,.22,'Conditional VAE\n4-class conditioning\n48-D latent code'),(.41,.57,.16,.22,'Hypothesis encoder\nEncode under all four\nclass hypotheses'),(.61,.57,.16,.22,'Adaptive classifier bank\nLR, SVM, Extra Trees,\nRF, deep MLP'),(.81,.57,.17,.22,'Calibration and triage\nTemperature scaling,\nentropy, referral'),(.21,.17,.16,.20,'Synthetic generator\nMinority-class latent\nsampling'),(.41,.17,.16,.20,'Quality gate\nCentroid distance and\nvalidation macro-F1'),(.61,.17,.16,.20,'Expert-system layer\nClass probabilities,\nrules, uncertainty'),(.81,.17,.17,.20,'Referral report\nPredicted class, risk tier,\nhuman-review trigger')]
for x,y0,w,h,t in items:
    ax.add_patch(FancyBboxPatch((x,y0),w,h,boxstyle='round,pad=.012',linewidth=1.25,facecolor='white')); ax.text(x+w/2,y0+h/2,t,ha='center',va='center',fontsize=9)
for x1,y1,x2,y2 in [(.17,.68,.21,.68),(.37,.68,.41,.68),(.57,.68,.61,.68),(.77,.68,.81,.68),(.29,.57,.29,.37),(.37,.27,.41,.27),(.49,.37,.49,.57),(.69,.57,.69,.37),(.77,.27,.81,.27),(.89,.57,.89,.37)]:
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=13,linewidth=1.2))
ax.text(.49,.45,'Accepted only when\nvalidation improves',ha='center',va='center',fontsize=8,style='italic')
ax.set_title('Proposed validation-gated Generative AI–augmented expert system',fontsize=13); plt.tight_layout(); plt.savefig(R/'fig_system_architecture.png',dpi=220,bbox_inches='tight'); plt.close()
# representative real image montage
def crop(path,size=220):
    a=np.fromfile(str(path),np.uint8); bgr=cv2.imdecode(a,cv2.IMREAD_COLOR); g=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY); rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    m=(g>12).astype(np.uint8)*255; n,l,s,c=cv2.connectedComponentsWithStats(m,8)
    if n>1:
        h,w=g.shape; vals=[]
        for i in range(1,n):
            x,y,ww,hh,area=s[i];cx,cy=c[i];pen=((cx-w/2)**2+(cy-h/2)**2)/(h*h+w*w);vals.append((area*(1-.6*pen),x,y,ww,hh))
        _,x,y,ww,hh=max(vals);pad=int(.03*max(ww,hh));x=max(0,x-pad);y=max(0,y-pad);rgb=rgb[y:min(h,y+hh+2*pad),x:min(w,x+ww+2*pad)]
    return ImageOps.fit(Image.fromarray(rgb),(size,size),Image.Resampling.LANCZOS)
root=P/'Dataset/Dataset/Training dataset/Otitis Media'; samples=[root/'AOM/aom (20).png',root/'ASOM/asom (19).JPG',root/'CSOM/csom (7).png',root/'Normal Tympanic Membrane/normal (34).png']
fig,axs=plt.subplots(1,4,figsize=(10.5,2.8))
for ax,path,c in zip(axs,samples,CL): ax.imshow(crop(path));ax.axis('off');ax.set_title(c)
fig.suptitle('Representative field-cropped otoscopic images from the expert-labelled cohort',fontsize=11);plt.tight_layout();plt.savefig(R/'fig_representative_images.png',dpi=220,bbox_inches='tight');plt.close()
# cVAE training
vh=pd.read_csv(R/'cvae_training_history.csv'); plt.figure(figsize=(6.4,4)); plt.plot(vh.epoch,vh.reconstruction,label='Reconstruction loss'); plt.plot(vh.epoch,vh.loss,label='Total loss'); plt.xlabel('Epoch');plt.ylabel('Loss');plt.title('Conditional VAE convergence');plt.legend();plt.tight_layout();plt.savefig(R/'fig_cvae_training.png',dpi=220);plt.close()
# model comparison
res=pd.read_csv(R/'same_split_model_comparison.csv'); d=res[res.regime=='GenAI-augmented'].sort_values('test_macro_f1'); x=np.arange(len(d));plt.figure(figsize=(8.5,4.6));plt.bar(x-.18,d.test_macro_f1,.36,label='Macro F1');plt.bar(x+.18,d.test_balanced_accuracy,.36,label='Balanced accuracy');plt.xticks(x,d.model,rotation=20,ha='right');plt.ylim(0,1);plt.ylabel('Score');plt.title('GenAI-regime candidate model comparison');plt.legend();plt.tight_layout();plt.savefig(R/'fig_model_comparison.png',dpi=220);plt.close()
# regime comparison for deep MLP
m=res[res.model=='Deep latent MLP'];x=np.arange(len(m));plt.figure(figsize=(7,4.2));plt.bar(x-.18,m.test_accuracy,.36,label='Accuracy');plt.bar(x+.18,m.test_macro_f1,.36,label='Macro F1');plt.xticks(x,m.regime,rotation=10);plt.ylim(0,1);plt.title('Effect of QA and generative augmentation on the deep latent MLP');plt.legend();plt.tight_layout();plt.savefig(R/'fig_regime_ablation.png',dpi=220);plt.close()
# confusion matrix
cm=confusion_matrix(y,pred,labels=np.arange(4));plt.figure(figsize=(5.5,4.6));plt.imshow(cm,cmap='Blues');plt.xticks(range(4),CL);plt.yticks(range(4),CL);plt.xlabel('Predicted');plt.ylabel('True');plt.title('Final calibrated ensemble confusion matrix')
for i in range(4):
 for j in range(4):plt.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
plt.colorbar();plt.tight_layout();plt.savefig(R/'fig_confusion_matrix.png',dpi=220);plt.close()
# ROC
plt.figure(figsize=(6.2,5));
for i,c in enumerate(CL):
 fpr,tpr,_=roc_curve((y==i).astype(int),p[:,i]);auc=roc_auc_score((y==i).astype(int),p[:,i]);plt.plot(fpr,tpr,label=f'{c} (AUC={auc:.3f})')
plt.plot([0,1],[0,1],'--',linewidth=1);plt.xlabel('False-positive rate');plt.ylabel('True-positive rate');plt.title('One-vs-rest ROC curves');plt.legend();plt.tight_layout();plt.savefig(R/'fig_roc_curves.png',dpi=220);plt.close()
# calibration
plt.figure(figsize=(6.2,5));
for i,c in enumerate(CL):
 frac,mean=calibration_curve((y==i).astype(int),p[:,i],n_bins=6,strategy='quantile');plt.plot(mean,frac,marker='o',label=c)
plt.plot([0,1],[0,1],'--');plt.xlabel('Mean predicted probability');plt.ylabel('Observed frequency');plt.title('Class-wise reliability diagram');plt.legend();plt.tight_layout();plt.savefig(R/'fig_calibration.png',dpi=220);plt.close()
# referral
s=pd.read_csv(R/'selective_referral_analysis.csv');plt.figure(figsize=(6.5,4.4));plt.plot(s.referral_rate,s.retained_accuracy,marker='o');plt.xlabel('Referral rate');plt.ylabel('Accuracy among retained automated cases');plt.ylim(0,1.02);plt.title('Risk–coverage trade-off for referral triage');plt.tight_layout();plt.savefig(R/'fig_selective_referral.png',dpi=220);plt.close()
# quality gate
q=pd.read_csv(R/'synthetic_quality_gate_grid.csv');q=q[q.model=='ExtraTrees'].sort_values('n_synth');plt.figure(figsize=(7,4.4));plt.scatter(q.n_synth,q.val_macro_f1,label='Validation macro F1');plt.scatter(q.n_synth,q.test_macro_f1,label='Test macro F1');plt.xlabel('Accepted synthetic images');plt.ylabel('Macro F1');plt.title('Synthetic quality-gate sensitivity');plt.legend();plt.tight_layout();plt.savefig(R/'fig_synthetic_gate.png',dpi=220);plt.close()
# box/violin
mets=['accuracy','balanced_accuracy','macro_f1','macro_auc','MCC'];vals=[b[c].dropna().values for c in mets];labs=['Accuracy','Balanced acc.','Macro F1','Macro AUC','MCC']
plt.figure(figsize=(8,4.5));plt.boxplot(vals,tick_labels=labs,showfliers=False);plt.ylim(0,1.02);plt.title('Bootstrap distributions of final test metrics');plt.tight_layout();plt.savefig(R/'fig_boxplot_bootstrap.png',dpi=220);plt.close()
plt.figure(figsize=(8,4.5));plt.violinplot(vals,showmedians=True,showextrema=True);plt.xticks(range(1,6),labs);plt.ylim(0,1.02);plt.title('Violin plots of bootstrap test metrics');plt.tight_layout();plt.savefig(R/'fig_violin_bootstrap.png',dpi=220);plt.close()
# deployment diagram
fig,ax=plt.subplots(figsize=(10,4.3));ax.axis('off');boxes=[(.03,.35,.15,.28,'Otoscope or\ntele-clinic upload'),(.23,.35,.15,.28,'CPU/GPU inference\nPyTorch 2.10\nPython 3.13'),(.43,.35,.15,.28,'Calibrated class\nprobabilities'),(.63,.35,.15,.28,'Uncertainty and\nreferral rules'),(.83,.35,.14,.28,'ENT review queue\nand report')]
for x,y0,w,h,t in boxes:ax.add_patch(FancyBboxPatch((x,y0),w,h,boxstyle='round,pad=.01',facecolor='white'));ax.text(x+w/2,y0+h/2,t,ha='center',va='center',fontsize=9)
for x in [.18,.38,.58,.78]:ax.add_patch(FancyArrowPatch((x,.49),(x+.05,.49),arrowstyle='-|>',mutation_scale=13))
ax.text(.5,.12,'Suggested deployment: Docker container, REST API, model/version registry, audit log, and human override',ha='center',fontsize=9);ax.set_title('Real-time deployment and referral workflow',fontsize=12);plt.tight_layout();plt.savefig(R/'fig_deployment.png',dpi=220,bbox_inches='tight');plt.close()
print('done',len(list(R.glob('fig_*.png'))))
