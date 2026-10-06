from __future__ import annotations

from pathlib import Path
import json, math, os, re, textwrap, zipfile, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch, Rectangle
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef,
    roc_auc_score, log_loss, brier_score_loss
)
from sklearn.preprocessing import label_binarize

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

BASE = Path('/mnt/data/eswa_final_work')
OUT = BASE / 'submission_package'
FIGDIR = OUT / 'figures'
RENDER = OUT / 'render_final'
OUT.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

REPRO = Path('/mnt/data/eswa_repro/ESWA_GenAI_QA724_Reproducibility_Package')
ROI_PRED = BASE / 'roi_results_fast/selected_dualview_predictions.npz'
GEN_PRED = REPRO / 'results/final_calibrated_predictions.npz'
STATS = json.loads((BASE / 'final_stats/final_statistics.json').read_text())
ROI_SUM = json.loads((BASE / 'roi_results_fast/selected_dualview_summary.json').read_text())
GEN_SUM = json.loads((REPRO / 'results/final_model_summary.json').read_text())
CLASSES = ['AOM','ASOM','CSOM','Normal']

roi = np.load(ROI_PRED)
gen = np.load(GEN_PRED)
y = roi['y']
proi = roi['probs']
pgen = gen['probs']
assert np.array_equal(y, gen['y'])
yp_roi = proi.argmax(1)
yp_gen = pgen.argmax(1)

# ---------- Plot helpers ----------
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.titlesize': 10.5,
    'axes.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 150,
})

BLUE = '#235789'; TEAL='#2A9D8F'; ORANGE='#F4A261'; RED='#D95D39'; PURPLE='#6C5B7B'; GRAY='#68717A'; LIGHT='#EFF4F8'; DARK='#17202A'; GREEN='#4C956C'


def savefig(fig, name, dpi=280):
    p = FIGDIR / name
    fig.savefig(p, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return p


def add_box(ax, xy, wh, title, sub='', fc='white', ec=BLUE, lw=1.3, fontsize=8.5, rounded=True):
    x,y0 = xy; w,h = wh
    patch = FancyBboxPatch((x,y0),w,h,boxstyle='round,pad=0.03,rounding_size=0.05' if rounded else 'square,pad=0.02',fc=fc,ec=ec,lw=lw)
    ax.add_patch(patch)
    ax.text(x+w/2,y0+h*0.63,title,ha='center',va='center',fontsize=fontsize,fontweight='bold',color=DARK)
    if sub:
        ax.text(x+w/2,y0+h*0.30,sub,ha='center',va='center',fontsize=fontsize-1.3,color=GRAY,wrap=True)
    return patch


def arr(ax, p1, p2, text=None, color=DARK, rad=0.0, ls='-', lw=1.15):
    a=FancyArrowPatch(p1,p2,arrowstyle='-|>',mutation_scale=9,connectionstyle=f'arc3,rad={rad}',lw=lw,ls=ls,color=color)
    ax.add_patch(a)
    if text:
        ax.text((p1[0]+p2[0])/2,(p1[1]+p2[1])/2+0.10,text,ha='center',va='bottom',fontsize=7,color=color)

# 0 Graphical abstract separate, 3.0:1-ish
fig, ax = plt.subplots(figsize=(13.28,5.31))
ax.set_xlim(0,13.28); ax.set_ylim(0,5.31); ax.axis('off')
ax.add_patch(Rectangle((0,0),13.28,5.31,fc='#FAFCFE',ec='none'))
ax.text(6.64,5.02,'Provenance-Gated Generative Evidence and Consensus Referral',ha='center',va='center',fontsize=16,fontweight='bold',color=DARK)
ax.text(6.64,4.68,'QA-724 otoscopic expert system: generated evidence is admitted only through validation and safety gates',ha='center',fontsize=10,color=GRAY)
xs=[0.35,2.75,5.15,7.55,9.95]
add_box(ax,(xs[0],2.70),(2.0,1.15),'QA firewall','699 expert images\n+ 204 review pool',fc='white',ec=BLUE)
add_box(ax,(xs[1],2.70),(2.0,1.15),'Two evidence views','cVAE hypothesis latent\n+ ROI morphology',fc='white',ec=PURPLE)
add_box(ax,(xs[2],2.70),(2.0,1.15),'Utility and fusion gate','Validation macro-F1\ncalibration + provenance',fc='white',ec=TEAL)
add_box(ax,(xs[3],2.70),(2.0,1.15),'Consensus referral','Agreement + confidence\n+ entropy',fc='white',ec=ORANGE)
add_box(ax,(xs[4],2.70),(2.95,1.15),'ENT-prioritized output','Class probabilities, referral tier,\naudit trail and human override',fc='white',ec=RED)
for i in range(4): arr(ax,(xs[i]+(2.0 if i<4 else 0),3.28),(xs[i+1],3.28))
# Results ribbon
metrics=[('Accuracy','78.8%'),('Macro F1','0.682'),('Macro AUC','0.870'),('MCC','0.598')]
for i,(k,v) in enumerate(metrics):
    x=0.55+i*2.05
    add_box(ax,(x,0.82),(1.72,1.10),v,k,fc='#F3F8FC',ec=BLUE,fontsize=10)
add_box(ax,(9.15,0.82),(3.55,1.10),'94.5% retained accuracy','47.1% referred; 86.4% of errors captured',fc='#FFF5EB',ec=ORANGE,fontsize=10)
ax.text(6.64,0.32,'Clinical boundary: image-level screening and referral support; not autonomous diagnosis',ha='center',fontsize=9.5,fontweight='bold',color=RED)
GRAPHICAL = savefig(fig,'graphical_abstract.png',dpi=300)

# Figure 1: provenance Sankey + class composition
fig, axes = plt.subplots(1,2,figsize=(10.8,4.2),gridspec_kw={'width_ratios':[1.4,1]})
ax=axes[0]; ax.set_xlim(0,12); ax.set_ylim(0,7); ax.axis('off')
ax.set_title('(a) Provenance and partition firewall',fontweight='bold')
add_box(ax,(0.3,4.65),(2.3,1.15),'Expert-labelled source','699 images',ec=BLUE)
add_box(ax,(0.3,1.55),(2.3,1.15),'Unlabelled review pool','204 images',ec=GRAY)
add_box(ax,(3.3,5.15),(2.4,1.0),'Cross-label conflicts','6 removed',ec=RED,fc='#FFF3F0')
add_box(ax,(3.3,3.65),(2.4,1.0),'Same-label copies','6 removed',ec=RED,fc='#FFF3F0')
add_box(ax,(3.3,1.25),(2.4,1.0),'Conservative candidates','37 accepted',ec=TEAL,fc='#F0FBF8')
add_box(ax,(6.3,4.15),(2.35,1.25),'Duplicate-clean core','687 expert images',ec=BLUE,fc='#F2F7FB')
add_box(ax,(9.25,4.15),(2.35,1.25),'QA-724 research set','687 expert + 37 pending',ec=PURPLE,fc='#F7F3FA')
add_box(ax,(9.25,1.30),(2.35,1.25),'Evaluation firewall','103 val + 104 test\nexpert only',ec=ORANGE,fc='#FFF7EF')
arr(ax,(2.6,5.22),(3.3,5.65)); arr(ax,(2.6,5.1),(3.3,4.15)); arr(ax,(2.6,5.22),(6.3,4.78),rad=-0.05)
arr(ax,(5.7,5.65),(6.3,5.05),color=RED); arr(ax,(5.7,4.15),(6.3,4.55),color=RED)
arr(ax,(2.6,2.12),(3.3,1.75)); arr(ax,(5.7,1.75),(9.25,4.45),color=TEAL,rad=-0.08)
arr(ax,(8.65,4.78),(9.25,4.78)); arr(ax,(8.65,4.35),(9.25,2.25),text='fixed expert split',color=ORANGE,rad=0.10)
ax.text(6.0,0.35,'Pseudo-labelled and generated samples never enter validation or testing.',ha='center',fontsize=8.2,fontweight='bold',color=RED)
ax=axes[1]
vals=[96,84,50,494]; labels=CLASSES; cols=[BLUE,RED,ORANGE,TEAL]
wedges,_=ax.pie(vals,startangle=90,colors=cols,wedgeprops={'width':0.42,'edgecolor':'white'})
ax.text(0,0.08,'QA-724',ha='center',va='center',fontsize=14,fontweight='bold')
ax.text(0,-0.14,'images',ha='center',va='center',fontsize=9,color=GRAY)
ax.legend(wedges,[f'{l}: {v}' for l,v in zip(labels,vals)],loc='lower center',bbox_to_anchor=(0.5,-0.22),ncol=2,frameon=False)
ax.set_title('(b) Research-set class composition',fontweight='bold')
fig.tight_layout()
FIG1=savefig(fig,'fig01_provenance_and_composition.png')

# Figure 2: representative images copy from archived
src_rep=REPRO/'figures/fig_representative_images.png'
FIG2=FIGDIR/'fig02_representative_otoscopy.png'; shutil.copy2(src_rep,FIG2)

# Figure 3: advanced multi-lane architecture
fig,ax=plt.subplots(figsize=(11.4,6.4)); ax.set_xlim(0,14); ax.set_ylim(0,8.7); ax.axis('off')
ax.text(7,8.42,'PG-GECR system architecture with validation, consensus, and clinical-feedback loops',ha='center',fontsize=14,fontweight='bold')
lanes=[(6.25,'DATA GOVERNANCE',BLUE),(3.75,'GENERATIVE AND MORPHOLOGICAL EVIDENCE',PURPLE),(1.25,'DECISION, REFERRAL, AND AUDIT',ORANGE)]
for yy,name,c in lanes:
    ax.add_patch(Rectangle((0.2,yy-0.3),13.6,1.85,fc='#FBFCFD',ec='#D7DEE5',lw=.8))
    ax.text(0.35,yy+1.28,name,fontsize=9,fontweight='bold',color=c)
# lane boxes
add_box(ax,(0.55,6.35),(2.1,0.95),'D1. Image intake','device, quality, provenance',ec=BLUE)
add_box(ax,(3.05,6.35),(2.1,0.95),'D2. QA audit','duplicates, formats, conflicts',ec=BLUE)
add_box(ax,(5.55,6.35),(2.1,0.95),'D3. Split firewall','expert-only val/test',ec=BLUE)
add_box(ax,(8.05,6.35),(2.1,0.95),'D4. QA-724 train','governed pseudo-labels',ec=BLUE)
add_box(ax,(10.55,6.35),(2.8,0.95),'D5. Versioned manifest','source, hash, label status',ec=BLUE)
for x in [2.65,5.15,7.65,10.15]: arr(ax,(x,6.83),(x+0.4,6.83))
add_box(ax,(0.55,3.85),(2.25,0.95),'G1. Conditional VAE','48-D class manifolds',ec=PURPLE)
add_box(ax,(3.15,3.85),(2.25,0.95),'G2. Four-hypothesis code','4 x 48-D = 192-D',ec=PURPLE)
add_box(ax,(5.75,3.85),(2.25,0.95),'G3. ROI evidence','color, texture, HOG, LBP',ec=TEAL)
add_box(ax,(8.35,3.85),(2.25,0.95),'G4. Utility gate','fidelity, distance, val F1',ec=TEAL)
add_box(ax,(10.95,3.85),(2.4,0.95),'G5. Evidence bank','latent + ROI probabilities',ec=PURPLE)
for x in [2.8,5.4,8.0,10.6]: arr(ax,(x,4.33),(x+0.35,4.33))
add_box(ax,(0.55,1.35),(2.25,0.95),'M1. Validation fusion','weights fixed before test',ec=ORANGE)
add_box(ax,(3.15,1.35),(2.25,0.95),'M2. Calibration','temperature, ECE, NLL',ec=ORANGE)
add_box(ax,(5.75,1.35),(2.25,0.95),'M3. Consensus gate','agreement + entropy',ec=ORANGE)
add_box(ax,(8.35,1.35),(2.25,0.95),'M4. Referral engine','retain, prioritize, repeat',ec=RED)
add_box(ax,(10.95,1.35),(2.4,0.95),'M5. ENT output','probabilities + rationale + log',ec=RED)
for x in [2.8,5.4,8.0,10.6]: arr(ax,(x,1.83),(x+0.35,1.83))
# cross-lane loops
arr(ax,(9.1,6.35),(1.65,4.80),'expert training stream',BLUE,rad=.15)
arr(ax,(11.9,3.85),(1.65,2.30),'candidate probabilities',PURPLE,rad=.10)
arr(ax,(6.85,1.35),(9.45,3.85),'reject / restrict synthetic set',RED,rad=.18,ls='--')
arr(ax,(12.1,1.35),(4.1,6.35),'ENT adjudication and drift feedback',RED,rad=-.24,ls='--')
ax.text(7,0.48,'Rule boundary: generated images are provisional evidence; validation decides admission and human review decides clinical action.',ha='center',fontsize=8.6,fontweight='bold',color=RED)
fig.tight_layout()
FIG3=savefig(fig,'fig03_advanced_system_architecture.png')

# Figure 4: cVAE convergence + synthetic montage + gate heatmap
hist=pd.read_csv(REPRO/'results/cvae_training_history.csv')
gate=pd.read_csv(REPRO/'results/synthetic_quality_gate_grid.csv')
fig=plt.figure(figsize=(10.8,8.0)); gs=fig.add_gridspec(2,2,height_ratios=[1,1.15])
ax=fig.add_subplot(gs[0,0]);
for c,label,col in [('reconstruction','Reconstruction',BLUE),('loss','Total',ORANGE)]:
    if c in hist.columns: ax.plot(hist['epoch'],hist[c],marker='o',ms=2.5,lw=1.5,label=label,color=col)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title('(a) Conditional VAE convergence',fontweight='bold'); ax.grid(alpha=.2); ax.legend(frameon=False)
# montage as image
ax=fig.add_subplot(gs[0,1]); ax.imshow(plt.imread(REPRO/'figures/synthetic_montage.png')); ax.axis('off'); ax.set_title('(b) Generated minority-class candidates',fontweight='bold')
# heatmap pivot for ExtraTrees validation macro F1 by synthetic counts; use sequence scatter matrix
ax=fig.add_subplot(gs[1,:])
g=gate[gate['model']=='ExtraTrees'].copy()
sc=ax.scatter(g['n_synth'],g['val_macro_f1'],c=g['test_macro_f1'],s=55+120*(g['val_macro_auc']-g['val_macro_auc'].min())/(g['val_macro_auc'].max()-g['val_macro_auc'].min()+1e-9),cmap='viridis',edgecolor='black',linewidth=.35)
best=g.loc[g['val_macro_f1'].idxmax()]
ax.scatter([best['n_synth']],[best['val_macro_f1']],s=180,marker='*',color=RED,label='Validation-selected setting')
ax.axhline(best['val_macro_f1'],ls='--',lw=.8,color=RED,alpha=.7)
ax.set_xlabel('Accepted generated images'); ax.set_ylabel('Validation macro F1'); ax.set_title('(c) Synthetic utility landscape: color = test macro F1, size = validation AUC',fontweight='bold'); ax.grid(alpha=.2)
cb=fig.colorbar(sc,ax=ax,pad=.01); cb.set_label('Test macro F1')
ax.legend(frameon=False,loc='lower right')
fig.tight_layout()
FIG4=savefig(fig,'fig04_cvae_generation_and_utility_gate.png')

# Figure 5: model comparison heatmap + ablation waterfall
cmp=pd.read_csv(REPRO/'results/same_split_model_comparison.csv')
# choose GenAI models + deep regimes + new branches
rows=[]
for _,r in cmp[cmp.regime.eq('GenAI-augmented')].iterrows():
    rows.append((r['model'],r['test_accuracy'],r['test_balanced_accuracy'],r['test_macro_f1'],r['test_macro_auc'],r['test_MCC']))
rows += [
    ('GenAI latent ensemble',STATS['genai_metrics']['accuracy'],STATS['genai_metrics']['balanced_accuracy'],STATS['genai_metrics']['macro_f1'],STATS['genai_metrics']['macro_auc'],STATS['genai_metrics']['MCC']),
    ('ROI expert branch',ROI_SUM['individual']['Expert ROI']['test']['accuracy'],ROI_SUM['individual']['Expert ROI']['test']['balanced_accuracy'],ROI_SUM['individual']['Expert ROI']['test']['macro_f1'],ROI_SUM['individual']['Expert ROI']['test']['macro_auc'],ROI_SUM['individual']['Expert ROI']['test']['MCC']),
    ('ROI QA branch',ROI_SUM['individual']['QA ROI']['test']['accuracy'],ROI_SUM['individual']['QA ROI']['test']['balanced_accuracy'],ROI_SUM['individual']['QA ROI']['test']['macro_f1'],ROI_SUM['individual']['QA ROI']['test']['macro_auc'],ROI_SUM['individual']['QA ROI']['test']['MCC']),
    ('Dual-view ROI fusion',STATS['roi_metrics']['accuracy'],STATS['roi_metrics']['balanced_accuracy'],STATS['roi_metrics']['macro_f1'],STATS['roi_metrics']['macro_auc'],STATS['roi_metrics']['MCC'])]
mdf=pd.DataFrame(rows,columns=['Model','Accuracy','Balanced accuracy','Macro F1','Macro AUC','MCC']).drop_duplicates('Model',keep='last')
fig,axes=plt.subplots(1,2,figsize=(11.2,5.4),gridspec_kw={'width_ratios':[1.55,1]})
ax=axes[0]; data=mdf.set_index('Model').values
im=ax.imshow(data,aspect='auto',cmap='YlGnBu',vmin=.35,vmax=.95)
ax.set_xticks(range(5),mdf.columns[1:],rotation=25,ha='right'); ax.set_yticks(range(len(mdf)),mdf['Model'])
for i in range(data.shape[0]):
    for j in range(data.shape[1]): ax.text(j,i,f'{data[i,j]:.3f}',ha='center',va='center',fontsize=7,color='black' if data[i,j]<.78 else 'white')
ax.set_title('(a) Same-split candidate and evidence-view comparison',fontweight='bold'); fig.colorbar(im,ax=ax,fraction=.025,pad=.02)
ax=axes[1]
base=STATS['genai_metrics']; final=STATS['roi_metrics']
metrics=['Accuracy','Balanced acc.','Macro F1','Macro AUC','MCC']
basev=[base['accuracy'],base['balanced_accuracy'],base['macro_f1'],base['macro_auc'],base['MCC']]
finalv=[final['accuracy'],final['balanced_accuracy'],final['macro_f1'],final['macro_auc'],final['MCC']]
x=np.arange(len(metrics)); w=.36
ax.bar(x-w/2,basev,w,label='Latent GenAI ensemble',color=PURPLE)
ax.bar(x+w/2,finalv,w,label='Dual-view ROI fusion',color=TEAL)
for i,(b,f) in enumerate(zip(basev,finalv)): ax.text(i+w/2,f+.012,f'+{f-b:.3f}',ha='center',fontsize=7,fontweight='bold')
ax.set_ylim(0,1); ax.set_xticks(x,metrics,rotation=30,ha='right'); ax.set_ylabel('Score'); ax.grid(axis='y',alpha=.2); ax.legend(frameon=False,fontsize=7)
ax.set_title('(b) Validation-selected evidence-view ablation',fontweight='bold')
fig.tight_layout()
FIG5=savefig(fig,'fig05_model_comparison_and_ablation.png')

# Figure 6: confusion matrix + ROC
fig,axes=plt.subplots(1,2,figsize=(10.8,4.5))
cm=confusion_matrix(y,yp_roi,labels=range(4)); ax=axes[0]
im=ax.imshow(cm,cmap='Blues')
ax.set_xticks(range(4),CLASSES,rotation=30,ha='right'); ax.set_yticks(range(4),CLASSES); ax.set_xlabel('Predicted class'); ax.set_ylabel('True class'); ax.set_title('(a) Expert-only test confusion matrix',fontweight='bold')
for i in range(4):
    for j in range(4):
        pct=cm[i,j]/max(cm[i].sum(),1)
        ax.text(j,i,f'{cm[i,j]}\n({pct:.0%})',ha='center',va='center',fontsize=8,color='white' if cm[i,j]>cm.max()/2 else 'black')
fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
ax=axes[1]; yb=label_binarize(y,classes=range(4))
for k,c in enumerate([BLUE,RED,ORANGE,TEAL]):
    fpr,tpr,_=roc_curve(yb[:,k],proi[:,k]); a=auc(fpr,tpr)
    ax.plot(fpr,tpr,lw=1.7,label=f'{CLASSES[k]} (AUC {a:.3f})',color=c)
ax.plot([0,1],[0,1],'--',color=GRAY,lw=.8); ax.set_xlabel('False-positive rate'); ax.set_ylabel('True-positive rate'); ax.set_title('(b) One-vs-rest ROC curves',fontweight='bold'); ax.grid(alpha=.2); ax.legend(frameon=False,loc='lower right')
fig.tight_layout()
FIG6=savefig(fig,'fig06_confusion_and_roc.png')

# Figure 7 calibration + binary decision curves
cal=pd.read_csv(BASE/'final_stats/calibration_bins.csv')
fig,axes=plt.subplots(1,2,figsize=(10.8,4.4))
ax=axes[0]
# robust columns
print('cal columns',cal.columns.tolist())
if {'mean_confidence','accuracy'}.issubset(cal.columns):
    ax.plot(cal['mean_confidence'],cal['accuracy'],marker='o',lw=1.6,color=BLUE,label='Observed')
elif {'confidence','empirical_accuracy'}.issubset(cal.columns):
    ax.plot(cal['confidence'],cal['empirical_accuracy'],marker='o',lw=1.6,color=BLUE,label='Observed')
else:
    numeric=cal.select_dtypes(include='number').columns
    ax.plot(cal[numeric[0]],cal[numeric[1]],marker='o',lw=1.6,color=BLUE,label='Observed')
ax.plot([0,1],[0,1],'--',color=GRAY,lw=.9,label='Ideal'); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.grid(alpha=.2); ax.set_xlabel('Mean confidence'); ax.set_ylabel('Empirical accuracy'); ax.set_title(f'(a) Reliability diagram (ECE {STATS["multiclass_ECE"]:.3f})',fontweight='bold'); ax.legend(frameon=False)
ax=axes[1]
# decision curve disease vs normal. net benefit across thresholds.
ybin=(y!=3).astype(int); pdisease=1-proi[:,3]; N=len(y)
ths=np.linspace(.05,.90,18); nb_model=[]; nb_all=[]
for t in ths:
    pred=pdisease>=t; tp=((pred==1)&(ybin==1)).sum(); fp=((pred==1)&(ybin==0)).sum()
    nb_model.append(tp/N - fp/N*(t/(1-t)))
    prev=ybin.mean(); nb_all.append(prev-(1-prev)*(t/(1-t)))
ax.plot(ths,nb_model,label='PG-GECR disease screen',color=TEAL,lw=1.8)
ax.plot(ths,nb_all,label='Refer all as disease',color=ORANGE,lw=1.2)
ax.axhline(0,color=GRAY,ls='--',lw=.9,label='Refer none')
ax.set_xlabel('Disease-probability threshold'); ax.set_ylabel('Net benefit'); ax.set_title('(b) Decision-curve analysis: disease vs Normal',fontweight='bold'); ax.grid(alpha=.2); ax.legend(frameon=False)
fig.tight_layout()
FIG7=savefig(fig,'fig07_calibration_and_decision_curve.png')

# Figure 8 consensus risk-coverage + error capture
cons=pd.read_csv(BASE/'final_stats/consensus_referral.csv')
fig,axes=plt.subplots(1,2,figsize=(10.8,4.4))
ax=axes[0]
# use columns returned
print('cons columns',cons.columns.tolist())
# standardize
if 'coverage' in cons.columns:
    cov=cons['coverage']; acc=cons['retained_accuracy']
else:
    cov=1-cons['referral_rate']; acc=cons['retained_accuracy']
sc=ax.scatter(cov,acc,c=cons['error_capture_rate'],s=70,cmap='plasma',edgecolor='black',linewidth=.4)
ax.plot(cov,acc,color=GRAY,lw=.8,alpha=.7); ax.set_xlabel('Automated coverage'); ax.set_ylabel('Retained-case accuracy'); ax.set_xlim(0,1); ax.set_ylim(.65,1.01); ax.grid(alpha=.2); ax.set_title('(a) Cross-view consensus risk-coverage frontier',fontweight='bold')
fig.colorbar(sc,ax=ax,pad=.01,label='Error capture')
# annotate chosen rule by nearest to 0.5288 coverage and .945 acc
idx=((cov-.528846).abs()+(acc-.94545).abs()).idxmin(); ax.annotate('Selected gate',xy=(cov.loc[idx],acc.loc[idx]),xytext=(cov.loc[idx]+.08,acc.loc[idx]-.08),arrowprops=dict(arrowstyle='->',color=RED),fontsize=8,fontweight='bold')
ax=axes[1]
# threshold matrix selected common rows
sel=cons.sort_values('retained_accuracy',ascending=False).head(8).sort_values('coverage' if 'coverage' in cons.columns else 'referral_rate')
x=np.arange(len(sel));
ax.bar(x,sel['retained_accuracy'],label='Retained accuracy',color=TEAL)
ax.plot(x,sel['error_capture_rate'],marker='o',color=RED,label='Error capture')
ax.set_ylim(0,1.05); ax.set_ylabel('Proportion'); ax.grid(axis='y',alpha=.2)
labels=[]
for _,r in sel.iterrows():
    if 'roi_confidence_threshold' in r: labels.append(f"{r['roi_confidence_threshold']:.2f}/0.50")
    else: labels.append(str(len(labels)+1))
ax.set_xticks(x,labels,rotation=45,ha='right'); ax.set_xlabel('ROI / generative confidence gate'); ax.set_title('(b) Consensus operating points',fontweight='bold'); ax.legend(frameon=False)
fig.tight_layout()
FIG8=savefig(fig,'fig08_consensus_referral.png')

# Figure 9 bootstrap forest + paired differences
boot=STATS['bootstrap']; dif=STATS['bootstrap_differences']
fig,axes=plt.subplots(1,2,figsize=(10.8,4.7))
ax=axes[0]
keys=['accuracy','balanced_accuracy','macro_f1','macro_auc','MCC','consensus_retained_accuracy']
labels=['Accuracy','Balanced accuracy','Macro F1','Macro AUC','MCC','Consensus retained accuracy']
ypos=np.arange(len(keys))[::-1]
for yy,k in zip(ypos,keys):
    d=boot[k]; ax.errorbar(d['median'],yy,xerr=[[d['median']-d['lower_95']],[d['upper_95']-d['median']]],fmt='o',color=BLUE,capsize=3)
ax.set_yticks(ypos,labels); ax.set_xlim(.25,1.02); ax.grid(axis='x',alpha=.2); ax.set_xlabel('Bootstrap estimate and 95% interval'); ax.set_title('(a) Robustness of final operating metrics',fontweight='bold')
ax=axes[1]
keys=['accuracy','balanced_accuracy','macro_f1','macro_auc','MCC','log_loss']; labels=['Accuracy','Balanced accuracy','Macro F1','Macro AUC','MCC','Log loss']
ypos=np.arange(len(keys))[::-1]
for yy,k in zip(ypos,keys):
    d=dif[k]; col=TEAL if (d['lower_95']>0 or d['upper_95']<0) else GRAY
    ax.errorbar(d['median'],yy,xerr=[[d['median']-d['lower_95']],[d['upper_95']-d['median']]],fmt='o',color=col,capsize=3)
ax.axvline(0,color=RED,ls='--',lw=.9); ax.set_yticks(ypos,labels); ax.grid(axis='x',alpha=.2); ax.set_xlabel('Paired bootstrap difference: ROI fusion - latent ensemble'); ax.set_title('(b) Evidence-view differences',fontweight='bold')
fig.tight_layout()
FIG9=savefig(fig,'fig09_bootstrap_forest.png')

# Figure 10 deployment sequence
fig,ax=plt.subplots(figsize=(11.2,5.7)); ax.set_xlim(0,14); ax.set_ylim(0,8); ax.axis('off')
actors=[('Capture\nclient',1),('API and\nquality gate',3.4),('ROI evidence\nservice',5.8),('Generative\nevidence service',8.2),('Consensus and\ncalibration',10.6),('ENT queue and\naudit registry',13)]
for name,x0 in actors:
    ax.add_patch(FancyBboxPatch((x0-.72,6.85),1.44,.65,boxstyle='round,pad=.03',fc=LIGHT,ec=BLUE,lw=1.1)); ax.text(x0,7.17,name,ha='center',va='center',fontsize=8,fontweight='bold')
    ax.plot([x0,x0],[.65,6.82],ls='--',lw=.8,color='#AAB4BE')
steps=[
(1,3.4,6.25,'1. upload image + metadata'),
(3.4,1,5.72,'2. quality / provenance response'),
(3.4,5.8,5.18,'3a. cropped ROI tensor'),
(3.4,8.2,4.65,'3b. four-hypothesis tensor'),
(5.8,10.6,4.10,'4a. ROI probability vector'),
(8.2,10.6,3.57,'4b. latent probability + entropy'),
(10.6,13,3.03,'5. calibrated class + referral rule'),
(13,10.6,2.50,'6. ENT decision / adjudication'),
(10.6,3.4,1.97,'7. audit log + model version'),
(3.4,1,1.43,'8. class probabilities + action tier')]
for x1,x2,y0,txt in steps:
    arr(ax,(x1,y0),(x2,y0),color=BLUE if x2>x1 else RED)
    ax.text((x1+x2)/2,y0+.12,txt,ha='center',fontsize=7.3,color=DARK)
ax.add_patch(Rectangle((2.65,.72),10.95,.43,fc='#FFF5EB',ec=ORANGE,lw=.9)); ax.text(8.12,.94,'Containerized inference | model registry | encrypted audit trail | human override | drift monitoring',ha='center',va='center',fontsize=8,fontweight='bold')
ax.set_title('Prospective deployment sequence with parallel evidence services and closed clinical feedback',fontsize=13,fontweight='bold')
fig.tight_layout()
FIG10=savefig(fig,'fig10_deployment_sequence.png')

# ---------- References ----------
# Numbered Elsevier-style list; source titles intentionally avoid the prohibited word stem.
REFS = [
"World Health Organization. World report on hearing. Geneva: World Health Organization; 2021.",
"Lieberthal AS, Carroll AE, Chonmaitree T, et al. The diagnosis and management of acute otitis media. Pediatrics. 2013;131:e964-e999. doi:10.1542/peds.2012-3488.",
"Schilder AGM, Chonmaitree T, Cripps AW, et al. Otitis media. Nat Rev Dis Primers. 2016;2:16063. doi:10.1038/nrdp.2016.63.",
"Rosenfeld RM, Shin JJ, Schwartz SR, et al. Clinical practice guideline: otitis media with effusion executive summary. Otolaryngol Head Neck Surg. 2016;154:201-214. doi:10.1177/0194599815624407.",
"Qureishi A, Lee Y, Belfield K, Birchall JP, Daniel M. Update on otitis media: prevention and treatment. Infect Drug Resist. 2014;7:15-24. doi:10.2147/IDR.S39637.",
"Singh AK, Gupta A, Agrawal A, Mehta R, Raghuvanshi AS. A deep learning-based method for detection of severity stages of otitis media by otoscopic images. Annu Int Conf IEEE Eng Med Biol Soc. 2025;2025:1-6. doi:10.1109/EMBC58623.2025.11251873.",
"Wu Z, Lin Z, Li L, et al. Deep learning for classification of pediatric otitis media. Laryngoscope. 2021;131:E2344-E2351. doi:10.1002/lary.29302.",
"Khan MA, Kwon S, Choo J, Hong SM, Kang SH, Park IH, Kim SK, Hong SJ. Automatic detection of tympanic membrane and middle ear infection from oto-endoscopic images via convolutional neural networks. Neural Netw. 2020;126:384-394. doi:10.1016/j.neunet.2020.03.023.",
"Sundgaard JV, Harte J, Bray P, et al. Deep metric learning for otitis media classification. Med Image Anal. 2021;71:102034. doi:10.1016/j.media.2021.102034.",
"Dubois C, Eigen D, Simon F, Couloigner V, Gormish M, Chalumeau M, Schmoll L, Cohen JF. Development and validation of a smartphone-based deep-learning-enabled system to detect middle-ear conditions in otoscopic images. NPJ Digit Med. 2024;7:162. doi:10.1038/s41746-024-01159-9.",
"Crowson MG, Hartnick CJ, Diercks GR, Gallagher TQ, Fracchia MS, Setlur J, Cohen MS. Machine learning for accurate intraoperative pediatric middle ear effusion diagnosis. Pediatrics. 2021;147(4):e2020034546. doi:10.1542/peds.2020-034546.",
"Cha D, Pae C, Seong SB, Choi JY, Park HJ. Automated diagnosis of ear disease using ensemble deep learning with a big otoendoscopy image database. EBioMedicine. 2019;45:606-614. doi:10.1016/j.ebiom.2019.06.050.",
"Habib AR, Xu Y, Bock K, et al. Evaluating the generalizability of deep learning image classification algorithms to detect middle ear disease using otoscopy. Sci Rep. 2023;13:5368. doi:10.1038/s41598-023-31921-0.",
"Zhong Z, Guo X, Jia D, Zheng H, Wu Z, Wang X. Artificial intelligence as an auxiliary tool in pediatric otitis media diagnosis. Int J Pediatr Otorhinolaryngol. 2024;187:112154. doi:10.1016/j.ijporl.2024.112154.",
"Sundgaard JV, Hannemose MR, Laugesen S, Bray P, Harte J, Kamide Y, Tanaka C, Paulsen RR, Christensen AN. Multi-modal deep learning for joint prediction of otitis media and diagnostic difficulty. Laryngoscope Investig Otolaryngol. 2024;9(1):e1199. doi:10.1002/lio2.1199.",
"Guo Q, Xie L, Zhou L. Deep learning for otitis media classification using otoscopic image. Medicine (Baltimore). 2025;104(48):e46218. doi:10.1097/MD.0000000000046218.",
"Seo HW, Ko DW, Oh J, Lee J, Ji YB, Han SY, Moon BI, Jeong JH, Chung JH. Development and validation of a CNN-based diagnostic pipeline for the diagnosis of otitis media. J Clin Med. 2025;14(23):8572. doi:10.3390/jcm14238572.",
"Chu YC, Chen YC, Hsu CY, Kuo CT, Cheng YF, Lin KH, Liao WH. Hybrid artificial intelligence frameworks for otoscopic diagnosis: integrating convolutional neural networks and large language models toward real-time mobile health. Digit Health. 2025;11:20552076251395449. doi:10.1177/20552076251395449.",
"Shie CK, Chang HT, Fan FC, Chen CJ, Fang TY, Wang PC. A hybrid feature-based segmentation and classification system for the computer aided self-diagnosis of otitis media. Annu Int Conf IEEE Eng Med Biol Soc. 2014;2014:4655-4658. doi:10.1109/EMBC.2014.6944662.",
"Senaras C, Moberly AC, Teknos T, Essig G, Elmaraghy C, Taj-Schaal N, Yu L, Gurcan MN. Detection of eardrum abnormalities using ensemble deep learning approaches. In: Mori K, Petrick N, editors. Medical Imaging 2018: Computer-Aided Diagnosis. Proc SPIE. 2018;10575:105751A. doi:10.1117/12.2293297.",
"He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. In: Proc IEEE CVPR. 2016:770-778. doi:10.1109/CVPR.2016.90.",
"Huang G, Liu Z, van der Maaten L, Weinberger KQ. Densely connected convolutional networks. In: Proc IEEE CVPR. 2017:4700-4708. doi:10.1109/CVPR.2017.243.",
"Tan M, Le QV. EfficientNet: rethinking model scaling for convolutional neural networks. In: Proc ICML. 2019:6105-6114.",
"Sandler M, Howard A, Zhu M, Zhmoginov A, Chen LC. MobileNetV2: inverted residuals and linear bottlenecks. In: Proc IEEE CVPR. 2018:4510-4520.",
"Kingma DP, Welling M. Auto-encoding variational Bayes. In: Proc ICLR. 2014.",
"Sohn K, Lee H, Yan X. Learning structured output representation using deep conditional generative models. In: Adv Neural Inf Process Syst. 2015;28:3483-3491.",
"Goodfellow I, Pouget-Abadie J, Mirza M, et al. Generative adversarial nets. Adv Neural Inf Process Syst. 2014;27:2672-2680.",
"Mirza M, Osindero S. Conditional generative adversarial nets. arXiv:1411.1784; 2014.",
"Ho J, Jain A, Abbeel P. Denoising diffusion probabilistic models. Adv Neural Inf Process Syst. 2020;33:6840-6851.",
"Rombach R, Blattmann A, Lorenz D, Esser P, Ommer B. High-resolution image synthesis with latent diffusion models. In: Proc IEEE CVPR. 2022:10684-10695.",
"Karras T, Aittala M, Hellsten J, Laine S, Lehtinen J, Aila T. Training generative adversarial networks with limited data. Adv Neural Inf Process Syst. 2020;33:12104-12114.",
"Frid-Adar M, Klang E, Amitai M, Goldberger J, Greenspan H. Synthetic data augmentation using GAN for liver lesion classification. Neurocomputing. 2018;321:321-331. doi:10.1016/j.neucom.2018.09.013.",
"Shin HC, Tenenholtz NA, Rogers JK, et al. Medical image synthesis for data augmentation and anonymization using generative adversarial networks. In: Simulation and Synthesis in Medical Imaging. Springer; 2018:1-11.",
"Koetzier LR, Wu J, Mastrodicasa D, et al. Generating synthetic data for medical imaging. Radiology. 2024;312:e232471. doi:10.1148/radiol.232471.",
"Khosravi B, Li F, Dapamede T, et al. Synthetically enhanced: unveiling synthetic data's potential in medical imaging research. eBioMedicine. 2024;104:105174. doi:10.1016/j.ebiom.2024.105174.",
"Brugnara G, Preetha CJ, Deike K, et al. Addressing the generalizability of AI in radiology using synthetic patient image data. Radiol Artif Intell. 2024;6:e230514. doi:10.1148/ryai.230514.",
"Han C, Hayashi H, Rundo L, et al. GAN-based synthetic brain MR image generation. In: Proc IEEE ISBI. 2018:734-738.",
"Shorten C, Khoshgoftaar TM. A survey on image data augmentation for deep learning. J Big Data. 2019;6:60. doi:10.1186/s40537-019-0197-0.",
"Chawla NV, Bowyer KW, Hall LO, Kegelmeyer WP. SMOTE: synthetic minority over-sampling technique. J Artif Intell Res. 2002;16:321-357. doi:10.1613/jair.953.",
"Dalal N, Triggs B. Histograms of oriented gradients for human detection. In: Proc IEEE CVPR. 2005:886-893. doi:10.1109/CVPR.2005.177.",
"Ojala T, Pietikainen M, Maenpaa T. Multiresolution gray-scale and rotation invariant texture classification with local binary patterns. IEEE Trans Pattern Anal Mach Intell. 2002;24:971-987. doi:10.1109/TPAMI.2002.1017623.",
"Cortes C, Vapnik V. Support-vector networks. Mach Learn. 1995;20:273-297. doi:10.1007/BF00994018.",
"Breiman L. Random forests. Mach Learn. 2001;45:5-32. doi:10.1023/A:1010933404324.",
"Geurts P, Ernst D, Wehenkel L. Extremely randomized trees. Mach Learn. 2006;63:3-42. doi:10.1007/s10994-006-6226-1.",
"Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. In: Proc ICML. 2017:1321-1330.",
"Lakshminarayanan B, Pritzel A, Blundell C. Simple and scalable predictive uncertainty estimation using deep ensembles. Adv Neural Inf Process Syst. 2017;30.",
"Gal Y, Ghahramani Z. Dropout as a Bayesian approximation: representing model uncertainty in deep learning. In: Proc ICML. 2016:1050-1059.",
"Ovadia Y, Fertig E, Ren J, et al. Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift. Adv Neural Inf Process Syst. 2019;32.",
"Geifman Y, El-Yaniv R. Selective classification for deep neural networks. Adv Neural Inf Process Syst. 2017;30.",
"DeVries T, Taylor GW. Learning confidence for out-of-distribution detection in neural networks. arXiv:1802.04865; 2018.",
"Selvaraju RR, Cogswell M, Das A, et al. Grad-CAM: visual explanations from deep networks via gradient-based localization. In: Proc IEEE ICCV. 2017:618-626. doi:10.1109/ICCV.2017.74.",
"Lundberg SM, Lee SI. A unified approach to interpreting model predictions. Adv Neural Inf Process Syst. 2017;30.",
"Sundararajan M, Taly A, Yan Q. Axiomatic attribution for deep networks. In: Proc ICML. 2017:3319-3328.",
"Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. 2024;385:e078378. doi:10.1136/bmj-2023-078378.",
"Mongan J, Moy L, Kahn CE Jr. Checklist for artificial intelligence in medical imaging (CLAIM): a guide for authors and reviewers. Radiol Artif Intell. 2020;2:e200029. doi:10.1148/ryai.2020200029.",
"Vasey B, Nagendran M, Campbell B, et al. Reporting guideline for the early-stage clinical evaluation of decision support systems driven by artificial intelligence: DECIDE-AI. BMJ. 2022;377:e070904. doi:10.1136/bmj-2022-070904.",
"Liu X, Rivera SC, Moher D, Calvert MJ, Denniston AK. Reporting guidelines for clinical trial reports for interventions involving artificial intelligence: CONSORT-AI extension. BMJ. 2020;370:m3164. doi:10.1136/bmj.m3164.",
"Rivera SC, Liu X, Chan AW, Denniston AK, Calvert MJ. Guidelines for clinical trial protocols for interventions involving artificial intelligence: SPIRIT-AI extension. BMJ. 2020;370:m3210. doi:10.1136/bmj.m3210.",
"Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: machine learning in Python. J Mach Learn Res. 2011;12:2825-2830.",
"Paszke A, Gross S, Massa F, et al. PyTorch: an imperative style, high-performance deep learning library. Adv Neural Inf Process Syst. 2019;32."
]
assert len(REFS)==60

# ---------- Word helpers ----------
def set_cell_margins(cell, top=55, start=75, bottom=55, end=75):
    tcPr=cell._tc.get_or_add_tcPr(); tcMar=tcPr.first_child_found_in('w:tcMar')
    if tcMar is None: tcMar=OxmlElement('w:tcMar'); tcPr.append(tcMar)
    for tag,val in [('top',top),('start',start),('bottom',bottom),('end',end)]:
        node=tcMar.find(qn(f'w:{tag}'))
        if node is None: node=OxmlElement(f'w:{tag}'); tcMar.append(node)
        node.set(qn('w:w'),str(val)); node.set(qn('w:type'),'dxa')

def set_cell_shading(cell, fill):
    tcPr=cell._tc.get_or_add_tcPr(); shd=tcPr.find(qn('w:shd'))
    if shd is None: shd=OxmlElement('w:shd'); tcPr.append(shd)
    shd.set(qn('w:fill'),fill)

def set_table_borders(table, color='808080', size='4'):
    tblPr=table._tbl.tblPr; borders=tblPr.first_child_found_in('w:tblBorders')
    if borders is None: borders=OxmlElement('w:tblBorders'); tblPr.append(borders)
    for edge in ('top','left','bottom','right','insideH','insideV'):
        el=borders.find(qn(f'w:{edge}'))
        if el is None: el=OxmlElement(f'w:{edge}'); borders.append(el)
        el.set(qn('w:val'),'single'); el.set(qn('w:sz'),size); el.set(qn('w:color'),color); el.set(qn('w:space'),'0')

def repeat_header(row):
    trPr=row._tr.get_or_add_trPr(); node=OxmlElement('w:tblHeader'); node.set(qn('w:val'),'true'); trPr.append(node)

def cant_split(row):
    trPr=row._tr.get_or_add_trPr(); node=OxmlElement('w:cantSplit'); trPr.append(node)

def add_page_field(p):
    p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    r=p.add_run(); begin=OxmlElement('w:fldChar'); begin.set(qn('w:fldCharType'),'begin')
    instr=OxmlElement('w:instrText'); instr.set(qn('xml:space'),'preserve'); instr.text=' PAGE '
    end=OxmlElement('w:fldChar'); end.set(qn('w:fldCharType'),'end')
    r._r.extend([begin,instr,end])

def keep_with_next(p):
    pPr=p._p.get_or_add_pPr(); n=pPr.find(qn('w:keepNext'))
    if n is None: n=OxmlElement('w:keepNext'); pPr.append(n)

def set_repeat_headers_font(run, name='Times New Roman'):
    run.font.name=name; run._element.rPr.rFonts.set(qn('w:eastAsia'),name)

def add_text(doc, text, style='Normal', bold_prefix=None):
    p=doc.add_paragraph(style=style)
    if bold_prefix and text.startswith(bold_prefix):
        r=p.add_run(bold_prefix); r.bold=True; p.add_run(text[len(bold_prefix):])
    else: p.add_run(text)
    return p

def add_bullets(doc, items):
    for item in items:
        p=doc.add_paragraph(style='Body Bullet')
        p.add_run(item)

def add_equation(doc, equation, number):
    p=doc.add_paragraph(style='Equation ESWA'); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.add_run(equation)
    r=p.add_run(f'    ({number})'); r.bold=True
    return p

def add_figure(doc, path, caption, width=6.72):
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before=Pt(2); p.paragraph_format.space_after=Pt(1)
    p.add_run().add_picture(str(path),width=Inches(width))
    c=doc.add_paragraph(caption,style='Caption ESWA'); c.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
    return c

def add_table(doc, headers, rows, widths=None, fontsize=8.2, caption=None):
    if caption:
        cp=doc.add_paragraph(caption,style='Table Caption ESWA'); keep_with_next(cp)
    table=doc.add_table(rows=1,cols=len(headers)); table.alignment=WD_TABLE_ALIGNMENT.CENTER; table.autofit=False
    if widths:
        for i,w in enumerate(widths): table.columns[i].width=Inches(w)
    for i,h in enumerate(headers):
        cell=table.rows[0].cells[i]; cell.text=str(h); set_cell_margins(cell); set_cell_shading(cell,'DCE6F1'); cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
        for p in cell.paragraphs:
            p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after=Pt(0)
            for r in p.runs: r.bold=True; r.font.name='Arial'; r.font.size=Pt(fontsize)
    repeat_header(table.rows[0]); cant_split(table.rows[0])
    for row in rows:
        cells=table.add_row().cells; cant_split(table.rows[-1])
        for i,v in enumerate(row):
            cells[i].text=str(v); set_cell_margins(cells[i]); cells[i].vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.TOP
            for p in cells[i].paragraphs:
                p.paragraph_format.space_after=Pt(0.5); p.paragraph_format.line_spacing=1.0
                p.alignment=WD_ALIGN_PARAGRAPH.LEFT if i==0 or len(str(v))>16 else WD_ALIGN_PARAGRAPH.CENTER
                for r in p.runs: r.font.name='Times New Roman'; r.font.size=Pt(fontsize)
    set_table_borders(table)
    return table

def setup_doc(header_text='PG-GECR otoscopic expert system'):
    doc=Document(); sec=doc.sections[0]
    sec.page_width=Inches(8.27); sec.page_height=Inches(11.69)
    sec.top_margin=Inches(.62); sec.bottom_margin=Inches(.58); sec.left_margin=Inches(.72); sec.right_margin=Inches(.72)
    sec.header_distance=Inches(.25); sec.footer_distance=Inches(.25)
    styles=doc.styles
    n=styles['Normal']; n.font.name='Times New Roman'; n._element.rPr.rFonts.set(qn('w:eastAsia'),'Times New Roman'); n.font.size=Pt(10.25)
    n.paragraph_format.line_spacing=1.03; n.paragraph_format.space_after=Pt(2.6); n.paragraph_format.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
    for nm,sz,bef,aft in [('Title',16.5,0,4),('Heading 1',12.7,7,2.3),('Heading 2',11.1,5,1.8),('Heading 3',10.3,4,1.5)]:
        s=styles[nm]; s.font.name='Arial'; s._element.rPr.rFonts.set(qn('w:eastAsia'),'Arial'); s.font.size=Pt(sz); s.font.bold=True
        s.paragraph_format.space_before=Pt(bef); s.paragraph_format.space_after=Pt(aft); s.paragraph_format.keep_with_next=True
    custom={
        'Abstract ESWA':(9.8,1.01,2.5,False), 'Caption ESWA':(8.3,1.0,2.4,True), 'Table Caption ESWA':(8.7,1.0,1.5,True),
        'Reference ESWA':(8.55,1.0,0.5,False), 'Equation ESWA':(10.1,1.0,2.0,False), 'Body Bullet':(9.8,1.01,1.6,False), 'Code ESWA':(7.8,1.0,1.0,False)
    }
    names=[s.name for s in styles]
    for nm,(sz,ls,after,ital) in custom.items():
        if nm not in names: s=styles.add_style(nm,WD_STYLE_TYPE.PARAGRAPH)
        else: s=styles[nm]
        s.font.name='Times New Roman' if nm!='Code ESWA' else 'Courier New'; s._element.rPr.rFonts.set(qn('w:eastAsia'),s.font.name); s.font.size=Pt(sz); s.font.italic=ital
        s.paragraph_format.line_spacing=ls; s.paragraph_format.space_after=Pt(after); s.paragraph_format.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
        if nm=='Body Bullet': s.paragraph_format.left_indent=Inches(.23); s.paragraph_format.first_line_indent=Inches(-.16)
        if nm=='Reference ESWA': s.paragraph_format.left_indent=Inches(.25); s.paragraph_format.first_line_indent=Inches(-.25)
    hp=sec.header.paragraphs[0]; hp.text=header_text; hp.alignment=WD_ALIGN_PARAGRAPH.CENTER
    for r in hp.runs: r.font.name='Arial'; r.font.size=Pt(7.8); r.font.italic=True; r.font.color.rgb=RGBColor(80,80,80)
    add_page_field(sec.footer.paragraphs[0])
    # Prevent widows/orphans where possible
    for sname in ['Normal','Abstract ESWA','Reference ESWA']:
        styles[sname].paragraph_format.widow_control=True
    return doc

# ---------- Manuscript content sections ----------
TITLE='A Generative AI-Augmented Deep Learning Expert System for Data-Efficient Detection and Referral Triage of Mucosal Middle-Ear Disease'
SHORT='PG-GECR GenAI expert system for otoscopic referral triage'

ABSTRACT=(
"Otoscopic artificial intelligence is often evaluated on small, imbalanced cohorts in which duplicate images, acquisition-source cues, and overconfident predictions can inflate apparent performance. This study presents a provenance-gated Generative AI expert system for four-class image-level screening of acute otitis media, acute suppurative otitis media, chronic suppurative otitis media, and normal tympanic membrane appearances. The source collection comprised 699 ENT-supervised images and 204 unlabelled review images. A duplicate audit removed six cross-label conflict images and six redundant copies, yielding 687 expert-labelled images. Thirty-seven unanimously predicted, high-confidence, non-duplicate images were admitted only to the training partition, producing the QA-724 research set; validation and testing remained expert-only. The proposed PG-GECR architecture integrates a conditional variational autoencoder, label-independent four-hypothesis latent encoding, an interpretable region-of-interest evidence branch, validation-utility gating, calibrated probability fusion, and cross-view consensus referral. Unrestricted synthetic balancing was rejected because its highest validation macro F1 occurred with zero admitted generated images. On 104 expert-labelled test images, the validation-selected dual-view ROI fusion achieved 0.788 accuracy, 0.688 balanced accuracy, 0.682 macro F1, 0.870 macro AUC, 0.598 Matthews correlation coefficient, 0.608 log loss, and 0.078 expected calibration error. Relative to the calibrated latent GenAI ensemble, paired bootstrap intervals supported gains in macro F1 and macro AUC, while McNemar testing did not establish a categorical accuracy difference (p=0.265). A cross-view consensus gate referred 47.1% of cases, retained 55 images at 94.5% accuracy, and captured 86.4% of errors. The system is positioned as a human-supervised referral aid, not an autonomous diagnostic device."
)

HIGHLIGHTS=[
'Provenance gates keep non-expert images outside validation and testing.',
'Four-hypothesis encoding supplies label-independent generative evidence.',
'Validation testing rejects synthetic candidates that lack clinical utility.',
'Dual-view fusion reaches 0.682 macro F1 and 0.870 macro AUC.',
'Consensus referral retains cases at 94.5% accuracy and captures 86.4% of errors.'
]

INTRO_PARAS=[
"Otitis media is a spectrum of inflammatory and infectious middle-ear disorders that may produce pain, fever, hearing difficulty, otorrhoea, tympanic membrane opacity, bulging, perforation, and chronic structural change. Timely recognition is clinically relevant because recurrent or persistent disease can affect hearing and developmental outcomes, particularly in children [1-5]. Diagnosis requires integration of symptoms, pneumatic otoscopy, tympanometry where available, and clinician interpretation. An image-only algorithm therefore addresses one component of the diagnostic pathway and should be framed as screening or referral support rather than a replacement for clinical examination.",
"Deep learning has shown strong capacity for otoscopic image classification [6-20]. Large internal datasets have produced high accuracies, and smartphone otoscopy has enabled prospective or remote testing. Nevertheless, direct comparison across studies is difficult because disease taxonomies, devices, image-selection rules, prevalence, patient-level grouping, and reference standards differ. Recent multicohort evidence has also shown that internal discrimination may not persist when acquisition sites change. Non-clinical features such as black borders, image hue, file format, resolution, and duplicated images can become shortcuts that are predictive inside a dataset but clinically meaningless outside it.",
"Generative AI is frequently proposed as a response to minority-class scarcity. Variational autoencoders, adversarial networks, diffusion models, and synthetic augmentation frameworks can generate additional examples or latent representations [25-39]. Yet generated images may reproduce source bias, memorize rare samples, collapse to a narrow visual mode, or omit subtle anatomy. In a small clinical cohort, visual plausibility alone is insufficient evidence that a generated image should be admitted to training. A responsible system must allow the generative component to be restricted or rejected when expert-only validation does not support its downstream utility.",
"A deployable expert system also requires explicit operational knowledge. Classifier outputs must be calibrated, uncertainty must be converted into a human-review action, data provenance must be traceable, and the model-selection rule must be fixed before the test set is opened [45-50]. Explainability methods can support model auditing [51-53], while reporting frameworks such as TRIPOD+AI, CLAIM, DECIDE-AI, CONSORT-AI, and SPIRIT-AI emphasize transparent cohort construction, reference standards, evaluation, and clinical boundaries [54-58]. These requirements motivate a system-level contribution rather than a single neural architecture.",
"This work develops a Provenance-Gated Generative Evidence and Consensus Referral (PG-GECR) expert system for otoscopic screening of mucosal middle-ear disease. Its novelty lies in the interaction of six auditable mechanisms: a provenance firewall; a conditional generative representation; four-hypothesis inference that avoids class-label leakage; a complementary ROI morphology branch; validation-based admission of pseudo-labelled or generated evidence; and a calibrated consensus referral controller. The objective is not to force GenAI into the final classifier. The objective is to use generative evidence only where it survives explicit tests of utility and safety."
]

CONTRIBS=[
"A duplicate- and provenance-aware reconstruction of the QA-724 research cohort from 699 expert-labelled images and 204 unlabelled review images.",
"A label-independent four-hypothesis cVAE representation in which every image is encoded under all class conditions, avoiding access to the true label at inference.",
"A formal utility gate that treats generated samples as provisional candidates and accepts them only when expert-only validation evidence supports admission.",
"A dual-view evidence design combining generative latent probabilities with ROI morphology probabilities derived from color, texture, local binary pattern, HOG, edge, sharpness, and low-resolution appearance features [40,41].",
"A cross-view consensus referral rule that uses agreement, calibrated confidence, and predictive entropy to trade automated coverage for retained-case accuracy.",
"A complete same-split evaluation with calibration, bootstrap intervals, paired evidence-view differences, McNemar testing, decision-curve analysis, latency profiling, and explicit clinical claim boundaries."
]

RELATED_PARAS=[
("2.1 Otoscopic artificial intelligence", "Early computer-aided otoscopy combined handcrafted color and texture descriptors with support-vector or ensemble classifiers. More recent work uses DenseNet, Xception, MobileNetV2, metric learning, residual networks, and multistep CNNs [6-24]. Khan et al. evaluated 2,484 oto-endoscopic images across Normal, chronic otitis media with perforation, and effusion, reporting approximately 95% accuracy and average AUROC near 0.99 [8]. Wu et al. trained on 10,703 pediatric images and reported 97.45% internal accuracy for Xception, with 90.66% accuracy on a prospective smartphone set [7]. These results demonstrate feasibility but also show that data volume, capture technology, and task definition strongly affect reported values."),
("2.2 Generalization, bias, and reference quality", "Sundgaard et al. used deep metric learning on 1,336 images and highlighted the value of structured latent representations [9]. Habib et al. examined cross-cohort generalization and showed that performance can decline when models are transferred across acquisition sources [13]. Other studies have reported consensus labelling, reader comparisons, or smartphone evaluation [10-18]. The literature therefore supports three design principles used here: duplicate removal before splitting, isolation of expert-only validation and testing, and explicit treatment of device and file-format associations."),
("2.3 Generative augmentation in medical imaging", "Conditional VAEs model a class-dependent latent distribution and can generate examples by sampling within a class manifold [25,26]. GAN and diffusion families can produce sharper images, although stability, memorization, and domain fidelity remain concerns [27-31]. Medical-image studies have reported gains from synthetic supplementation in selected settings [32-37], while broad surveys emphasize that downstream task utility and external evaluation are essential [34-39]. PG-GECR therefore uses the cVAE for representation and candidate generation but does not assume that synthetic volume is beneficial."),
("2.4 Calibration and selective referral", "Modern neural networks can produce probability estimates that are too sharp [45]. Deep ensembles, Monte Carlo dropout, and dataset-shift analyses provide complementary uncertainty perspectives [46-48]. Selective classification formalizes the option to abstain on uncertain cases [49], and confidence learning can flag out-of-distribution inputs [50]. In this study, temperature scaling is combined with normalized entropy and cross-view agreement so that uncertainty has an operational consequence: retention, priority referral, or repeat acquisition."),
("2.5 Explainability and reporting", "Grad-CAM, SHAP, and integrated gradients can localize or attribute model decisions [51-53]. The present system additionally uses an interpretable ROI evidence stream whose feature groups can be audited directly. Reporting follows the structure advocated by TRIPOD+AI and CLAIM, while prospective translation is discussed against DECIDE-AI and the AI trial extensions [54-58]. Python implementations use scikit-learn and PyTorch [59,60].")
]

PUBLISHED_ROWS=[
('Khan et al. [8]','2020','2,484','Normal/COM/OME','DenseNet-family CNN','Accuracy 95%; AUROC ~0.99','Single-source internal evaluation'),
('Wu et al. [7]','2021','12,203 + 102 prospective','AOM/OME/Normal','Xception; MobileNetV2','97.45% internal; 90.66% smartphone','Large cohort; task differs from current four-class study'),
('Sundgaard et al. [9]','2021','1,336','AOM/OME/no effusion','Deep metric learning','Comparable to clinical experts','Metric-learning focus'),
('Crowson et al. [11]','2021','Small intraoperative cohort','Effusion present/absent','Neural classifier','Accuracy 83.8%; AUC 0.93; F1 0.80','Binary intraoperative task'),
('Dubois et al. [10]','2024','45,606 development + 326 test','11 middle-ear conditions','Smartphone deep-learning system','Normal/abnormal sensitivity 99.0%; specificity 95.2%','Held-out test and device-oriented workflow'),
('Cha et al. [12]','2019','10,544','Six ear-disease classes','InceptionV3 + ResNet101 ensemble','Five-fold accuracy 93.67%','Large internal otoendoscopy database'),
('Habib et al. [13]','2023','Multiple cohorts','Middle-ear disease','Cross-cohort CNN analysis','Internal-to-external decline','Generalization study'),
('Zhong et al. [14]','2024','Pediatric cohort','AOM/OME/Normal','CNN decision support','Reported clinical feasibility','Single-center evaluation'),
('Guo et al. [16]','2025','819','Otitis classes','VGG19-based classifier','Reported accuracy 94.51%','Consensus-labelled internal cohort'),
('Current PG-GECR','2026','724 research images; 104 expert test','AOM/ASOM/CSOM/Normal','GenAI + ROI evidence + referral','Macro F1 0.682; AUC 0.870; referral retained acc. 0.945','Duplicate-clean expert-only test; no external validation')
]

# Methods text
METHODS_SECTIONS = [
('3.1 Study design and clinical scope',
"This is a retrospective image-level development and evaluation study. The input is an RGB otoscopic or otoendoscopic image. The output is a calibrated four-class probability vector, a selected image-level class, an uncertainty score, and a referral action. Clinical symptoms, tympanic membrane mobility, tympanometry, audiometry, and patient history were unavailable. Accordingly, the model output is not a patient diagnosis and does not provide a validated complication-risk score."),
('3.2 Source cohort and ground-truth hierarchy',
"The source training directory contained 699 ENT-supervised images: 96 AOM, 90 ASOM, 48 CSOM, and 465 Normal. A separate directory contained 204 images without expert reference labels. Three evidence levels were retained in the manifest: expert-labelled images; conservative pseudo-labelled images pending ENT adjudication; and unaccepted review-pool predictions. Only the expert-labelled level was eligible for validation and test evaluation."),
('3.3 Duplicate and acquisition-source audit',
"Perceptual-hash and filename-normalization analyses identified three AOM-versus-Normal conflict pairs. All six participating images were removed because neither member could be selected without adjudication. Six further redundant copies from same-label pairs were removed. The resulting expert core contained 687 images. The audit also identified a strong association between JPG acquisition and the ASOM source class. JPG candidates from the unlabelled pool were therefore excluded from pseudo-label admission to reduce source-format shortcut risk."),
('3.4 Conservative pseudo-label admission',
"A four-model screening ensemble used field-cropped color, texture, local binary pattern, HOG, edge, and sharpness descriptors. An unlabelled image was admitted only if it was readable, stored in the non-confounded PNG domain, predicted identically by logistic regression, RBF-SVM, Extra Trees, and LightGBM, exceeded class-specific confidence and top-two-margin thresholds, and was not a duplicate of training or another candidate. Thirty-seven images passed: 3 AOM, 2 CSOM, and 32 Normal. These images were appended only to training and remain pending ENT confirmation."),
('3.5 Fixed data partition and leakage control',
"The duplicate-clean expert core was stratified with random seed 42 into 480 training, 103 validation, and 104 test images. The 37 pseudo-labelled additions were appended only to the 480-image training partition. The validation and test sets were identical across all regimes. Hyperparameters, branch weights, synthetic admission, calibration temperature, and referral thresholds were selected without access to test labels."),
('3.6 Image preprocessing and ROI representation',
"Images were decoded with OpenCV, converted to RGB, and cropped to the largest central non-black field with 3% padding. The cVAE used 64 x 64 RGB inputs scaled to [0,1]. The ROI branch generated 2,381 standardized features from RGB, HSV, and Lab moments and histograms; multi-radius local binary patterns; HOG; edge density; sharpness; and a low-resolution thumbnail. This branch captures interpretable morphology and acquisition appearance while limiting black-border and resolution cues."),
]

# Architecture section text
ARCH_SECTIONS=[
('4.1 PG-GECR knowledge rules',
"The expert-system layer enforces seven rules. R1: non-expert images cannot enter validation or testing. R2: duplicate conflicts are removed before splitting. R3: every conditional image is encoded under all four class hypotheses. R4: generated candidates are admitted only when expert-only validation supports the utility rule. R5: evidence-view weights and calibration parameters are learned on validation data only. R6: low-confidence, high-entropy, or cross-view disagreement cases are referred. R7: all outputs remain subordinate to ENT review and clinical context."),
('4.2 Conditional variational autoencoder',
"The cVAE encoder contains four stride-2 convolutional blocks with 24, 48, 96, and 128 channels, batch normalization, and leaky-ReLU activations. The flattened representation is projected to a 48-dimensional mean and log-variance. The decoder concatenates the sampled latent vector with a one-hot class code and uses transposed convolutions to reconstruct a 64 x 64 RGB image. A linearly scheduled KL term controls posterior regularization."),
('4.3 Label-independent four-hypothesis encoding',
"A standard conditional encoder requires a class input that is unknown at inference. PG-GECR resolves this by encoding each image under AOM, ASOM, CSOM, and Normal hypotheses. Concatenating the four latent means yields a 192-dimensional representation. The representation expresses compatibility with each conditional manifold without revealing the true class."),
('4.4 Generated-candidate utility gate',
"For each disease class, candidate latent vectors were sampled around the expert-training class prototype and clipped to a bounded standard-deviation region. Candidate admission was evaluated as a constrained validation problem. The primary term was change in validation macro F1, followed by balanced accuracy and log loss. Additional penalties represented centroid distance, near-duplicate risk, and class-distribution distortion. The null action - accepting no generated images - remained an eligible solution."),
('4.5 Dual-view ROI evidence fusion',
"The ROI branch contains two independently trained, class-balanced multinomial logistic models: an expert-only model and a QA-training model that includes the 37 conservative pseudo-labels. Candidate regularization values were selected on the fixed validation set. A validation sweep selected a 0.60 expert / 0.40 QA probability fusion. This fusion is deliberately separate from the cVAE latent ensemble, permitting cross-view comparison and consensus referral."),
('4.6 Calibration and cross-view consensus referral',
"Temperature scaling was fitted on validation probabilities. Normalized entropy was used as an uncertainty score. The deployment controller retains a case only when the ROI and generative branches agree, the ROI confidence is at least 0.70, the latent GenAI confidence is at least 0.50, ROI entropy is at most 0.65, and latent entropy is at most 0.75. All other cases are sent to the review queue. The rule was selected from validation behavior and examined on the expert-only test set."),
('4.7 Computational form and traceability',
"Every output record contains source hash, preprocessing version, model versions, class probabilities from both views, calibrated confidence, entropy, branch agreement, and referral action. This structure supports audit, rollback, threshold revision, and later replacement of pseudo-labels by adjudicated ENT labels. The modular implementation also allows a higher-resolution diffusion generator to replace the cVAE without altering the evaluation firewall or referral controller."),
]

# Experiments
EXP_SECTIONS=[
('5.1 Execution environment',
"Experiments were executed in a CPU-only Linux container using Python 3.13.5, PyTorch 2.10.0+cpu, torchvision 0.25.0+cpu, scikit-learn, OpenCV, NumPy, pandas, SciPy, Pillow, and Matplotlib. Python, NumPy, PyTorch, and scikit-learn seeds were fixed. Dataset manifests, feature arrays, predictions, bootstrap samples, figures, and metric files were archived with the code."),
('5.2 Training regimes',
"Four regimes were evaluated: expert-only training; QA-724 training with 37 conservative pseudo-labels; unrestricted cVAE balancing with 198 generated disease candidates; and validation-gated generation. The cVAE was trained only on the expert training partition. The dual-view ROI fusion used expert and QA variants but did not admit generated images because the synthetic utility gate selected the null candidate set."),
('5.3 Classifier bank and selection',
"The latent branch compared multinomial logistic regression, calibrated RBF-SVM, Extra Trees, Random Forest, and a deep latent multilayer perceptron. The ROI branch used class-balanced multinomial logistic regression because validation experiments showed that strong regularization and probability averaging were stable for the available sample size. Model ranking used validation macro F1 as the primary criterion, balanced accuracy as the secondary criterion, and macro AUC and log loss as tie-breakers."),
('5.4 Evaluation metrics and statistical analysis',
"Primary metrics were macro F1 and macro one-vs-rest AUC. Secondary metrics were accuracy, balanced accuracy, macro precision, macro recall, multiclass MCC, log loss, expected calibration error, and multiclass Brier score. Class-wise precision, recall, F1, support, and AUC were reported. Two thousand nonparametric bootstrap replicates estimated percentile intervals. Paired bootstrap differences compared the ROI fusion with the latent GenAI ensemble on the same test images. McNemar's exact test compared discordant correct/incorrect decisions. A binary disease-versus-Normal analysis reported AUC, average precision, Brier score, and decision curves."),
('5.5 Referral analysis and computational profiling',
"Referral evaluation reported automated coverage, referral rate, retained-case accuracy, and error capture. Coverage is the proportion of test images that pass the consensus gate; error capture is the fraction of all model errors assigned to review. Feature extraction was profiled serially on 40 images, and batch probability inference was profiled on 104 test representations. These measurements characterize the archived container and are not claims about prospective clinical latency."),
]

# Results paragraphs
RESULT_SECTIONS=[
('6.1 Cohort governance and class distribution',
"The QA audit reduced the expert-labelled cohort from 699 to 687 images before partitioning. The QA-724 development set contained 96 AOM, 84 ASOM, 50 CSOM, and 494 Normal images. The Normal class represented 68.2% of the research set. The 37 pseudo-labels were confined to training, and no pseudo-labelled or generated image entered the 103-image validation or 104-image test set."),
('6.2 Generative training and synthetic utility',
"The cVAE reconstruction loss reached approximately 0.0061 after 18 epochs. Prototype sampling produced 52 AOM, 61 ASOM, and 85 CSOM candidates. Generated images preserved coarse field shape and color distribution but showed variable fine anatomy. Across the synthetic-gate grid, the largest validation macro F1 was obtained with zero accepted generated images. Limited CSOM synthesis changed some test metrics, but it did not satisfy the pre-specified validation rule. The deployment training set therefore excluded generated images."),
('6.3 Candidate-model and evidence-view comparison',
"Within the unrestricted GenAI regime, logistic regression produced the largest test macro F1 among individual latent classifiers, while Extra Trees produced the strongest validation macro F1. The calibrated latent ensemble achieved 0.721 accuracy, 0.556 balanced accuracy, 0.532 macro F1, 0.785 macro AUC, and 0.469 MCC. The validation-selected dual-view ROI fusion achieved 0.788 accuracy, 0.688 balanced accuracy, 0.682 macro F1, 0.870 macro AUC, and 0.598 MCC. Log loss decreased from 0.779 to 0.608."),
('6.4 Paired statistical comparison',
"The point differences for ROI fusion minus latent ensemble were +0.067 accuracy, +0.132 balanced accuracy, +0.150 macro F1, +0.085 macro AUC, +0.129 MCC, and -0.171 log loss. Paired bootstrap 95% intervals were 0.010 to 0.280 for macro F1, 0.006 to 0.163 for macro AUC, and -0.344 to -0.009 for log loss. Accuracy and MCC intervals included zero. McNemar's exact test counted 18 images correct only under ROI fusion and 11 correct only under the latent ensemble; p=0.265. The evidence supports stronger probability ranking and macro F1 for the ROI stream, but not a definitive categorical accuracy difference in this test cohort."),
('6.5 Class-wise performance and error structure',
"The final ROI fusion correctly classified 10 of 14 AOM, 12 of 13 ASOM, 2 of 7 CSOM, and 58 of 70 Normal images. AOM recall was 0.714 and ASOM recall was 0.923. CSOM recall remained 0.286, reflecting low support and heterogeneous chronic morphology. Class AUC values were 0.859 for AOM, 0.996 for ASOM, 0.775 for CSOM, and 0.849 for Normal. The nonzero CSOM signal is stronger than the latent ensemble's argmax result, but the class remains unsuitable for autonomous classification."),
('6.6 Calibration and binary screening utility',
"Temperature scaling selected T=0.888 for the dual-view ROI probabilities. Test log loss was 0.608, multiclass expected calibration error was 0.078, and multiclass Brier score was 0.309. In the secondary disease-versus-Normal analysis, AUC was 0.849, average precision 0.822, and Brier score 0.153. At a 0.50 disease threshold, accuracy was 0.750, sensitivity 0.706, specificity 0.771, and F1 0.649. Decision curves showed positive net benefit over a range of clinically plausible thresholds, subject to external calibration."),
('6.7 Consensus referral performance',
"The selected cross-view gate retained 55 of 104 test images and referred 49, corresponding to 52.9% automated coverage and 47.1% referral. Retained-case accuracy was 0.945, and 86.4% of all errors were captured in the referral queue. Bootstrap medians were 0.529 for coverage, 0.947 for retained accuracy, and 0.870 for error capture; respective 95% intervals were 0.433-0.625, 0.878-1.000, and 0.696-1.000. Agreement alone retained more cases but produced lower retained accuracy, showing that confidence and entropy constraints contributed operational value."),
('6.8 Robustness and runtime',
"For the final ROI fusion, bootstrap 95% intervals were 0.712-0.865 for accuracy, 0.570-0.810 for balanced accuracy, 0.567-0.790 for macro F1, 0.784-0.944 for macro AUC, and 0.436-0.742 for MCC. The wider minority-sensitive intervals reflect only seven CSOM test images. Serial ROI feature extraction averaged 27.7 ms per image in the archived CPU container. Batch probability generation for 104 feature vectors averaged 5.5 ms in total. Image decoding, network transfer, and clinical platform latency were not included."),
]

DISCUSSION_PARAS=[
"The central result is that GenAI contributed most value as an auditable evidence and rejection mechanism rather than as a source of unlimited synthetic training volume. The conditional latent branch created a distinct representation and an independent probability stream. However, generated candidates did not survive the expert-only validation criterion. The system retained the generative branch for cross-view uncertainty and referral while selecting the ROI fusion as the primary class-probability stream. This is a substantive expert-system behavior: the architecture can decline evidence from one of its own modules.",
"Published otoscopic studies often report higher internal accuracy than the current four-class test result [7-18]. Those studies frequently use larger cohorts, binary or three-class tasks, different disease definitions, or less stringent source separation. The current experiment removed known duplicate conflicts, preserved a difficult chronic class, and prohibited non-expert samples from evaluation. The resulting estimates should therefore be interpreted as a conservative same-center development study rather than a direct ranking against published models.",
"The paired evidence-view analysis clarifies where the gain arose. ROI fusion increased macro F1 and macro AUC relative to the latent ensemble, and reduced log loss. The conditional latent model remained useful because disagreement between the generative and morphology views concentrated errors. Cross-view consensus converted that disagreement into a referral action and retained a subset with 94.5% accuracy. The selected gate therefore couples performance and workflow rather than optimizing accuracy in isolation.",
"Clinical utility is most plausible in tele-otoscopy or primary-care queue management. A technician could upload an image, receive a quality flag, calibrated probabilities, and a referral tier, and submit uncertain or disease-positive cases for ENT review. A repeat image can be requested when both confidence and agreement are low. Symptoms, age, fever, discharge duration, hearing status, tympanic membrane mobility, and tympanometry must override or contextualize the image output. No case should be discharged solely on a high Normal probability.",
"The study has several limitations. It is single-center and image-level; patient identifiers were unavailable, so patient-level grouping could not be verified beyond duplicate auditing. The 37 pseudo-labels await independent two-ENT adjudication. The source ASOM class is associated with JPG acquisition. The cVAE used 64 x 64 inputs and did not consistently reproduce fine anatomy. CSOM support is small, and the test recall is inadequate for stand-alone use. No external site, device, operator, prospective, or reader-study validation was available. Referral thresholds were selected on internal validation and require prospective workload calibration.",
"The next phase should obtain consensus labels from two ENT specialists with adjudication, record patient and device identifiers, collect balanced chronic and difficult Normal mimics, and use grouped patient-level cross-validation. A higher-resolution conditional diffusion model can be compared with the cVAE only after memorization and anatomical-fidelity audits. External evaluation should report site- and device-stratified discrimination, calibration, decision curves, subgroup performance, referral workload, and reader comparison under DECIDE-AI principles [54-58]."
]

CONCLUSION=(
"This study presents PG-GECR, a provenance-gated Generative AI expert system for data-efficient otoscopic screening and referral triage. The framework reconstructs a duplicate-clean QA-724 research cohort, protects expert-only validation and testing, combines conditional latent and ROI morphology evidence, tests generated candidates through a null-eligible utility gate, calibrates probabilities, and converts cross-view uncertainty into human referral. The validation-selected ROI fusion achieved 0.682 macro F1 and 0.870 macro AUC on 104 expert-labelled test images. Generated balancing was rejected, while the retained generative branch supported consensus referral. Referring 47.1% of cases yielded 94.5% accuracy among retained images and captured 86.4% of errors. These findings support a cautious human-supervised prioritization role. ENT adjudication, balanced multicenter data, external calibration, and prospective workflow evaluation are mandatory before clinical translation."
)

# Tables data
SPLIT_ROWS=[
('Expert training','65','59','33','323','480','Expert-labelled'),
('Validation','14','13','7','69','103','Expert-labelled only'),
('Test','14','13','7','70','104','Expert-labelled only'),
('Pseudo-label additions','3','0','2','32','37','Training only; pending ENT'),
('QA training total','68','59','35','355','517','Expert + conservative pending')]

RULE_ROWS=[
('R1','Evaluation firewall','Only expert-labelled images enter validation and testing.'),
('R2','Provenance audit','Remove duplicate conflicts and same-label copies before splitting.'),
('R3','Hypothesis inference','Encode each image under all four class conditions.'),
('R4','Generated-evidence utility','Accept generated candidates only when validation criteria support admission.'),
('R5','Model selection','Fix evidence weights and calibration on validation data only.'),
('R6','Referral','Refer disagreement, low confidence, high entropy, or poor quality.'),
('R7','Clinical boundary','Use output for screening and prioritization under ENT oversight.')]

HYPER_ROWS=[
('Input crop','Largest central non-black field; 3% padding'),('cVAE input','64 x 64 x 3; scaled to [0,1]'),('cVAE channels','24, 48, 96, 128'),('Latent dimension','48 per hypothesis; 192 concatenated'),('Optimizer','AdamW; learning rate 1.5e-3; weight decay 1e-5'),('cVAE schedule','18 epochs; KL coefficient 0.00065 to 0.003; gradient clip 5.0'),('Prototype sampling','0.55 x class latent SD; clip +/-1.5 SD'),('Generated candidates','52 AOM; 61 ASOM; 85 CSOM'),('ROI features','2,381 color, texture, HOG, LBP, edge, sharpness, thumbnail'),('ROI logistic C','Expert 0.05; QA 0.03'),('ROI evidence weights','0.60 expert / 0.40 QA'),('Temperature','0.888 final ROI; 0.960 latent ensemble'),('Consensus gate','ROI p>=0.70; GenAI p>=0.50; entropies <=0.65/0.75; agreement'),('Bootstrap','2,000 paired replicates; seed fixed')]

MODEL_ROWS=[]
for _,r in mdf.iterrows():
    MODEL_ROWS.append((r['Model'],f"{r['Accuracy']:.3f}",f"{r['Balanced accuracy']:.3f}",f"{r['Macro F1']:.3f}",f"{r['Macro AUC']:.3f}",f"{r['MCC']:.3f}"))

DELTA_ROWS=[]
for k,label in [('accuracy','Accuracy'),('balanced_accuracy','Balanced accuracy'),('macro_f1','Macro F1'),('macro_auc','Macro AUC'),('MCC','MCC'),('log_loss','Log loss')]:
    d=STATS['bootstrap_differences'][k]
    DELTA_ROWS.append((label,f"{d['point']:+.3f}",f"{d['lower_95']:+.3f} to {d['upper_95']:+.3f}",'Supported' if (d['lower_95']>0 or d['upper_95']<0) else 'Interval includes zero'))

class_rep=pd.read_csv(BASE/'final_stats/final_classification_report.csv')
CLASS_ROWS=[]
for _,r in class_rep.iterrows():
    if str(r.iloc[0]) in CLASSES:
        label=str(r.iloc[0]);
        # columns likely label, precision, recall, f1-score, support
        vals=r.to_dict();
        precision=vals.get('precision',r.iloc[1]); recall=vals.get('recall',r.iloc[2]); f1=vals.get('f1-score',vals.get('f1',r.iloc[3])); support=vals.get('support',r.iloc[4])
        CLASS_ROWS.append((label,f'{float(precision):.3f}',f'{float(recall):.3f}',f'{float(f1):.3f}',str(int(float(support))),f"{STATS['class_auc'][label]:.3f}"))

REFERRAL_ROWS=[
('Agreement only','70.2%','29.8%','87.7%','59.1%'),
('ROI 0.50 / GenAI 0.60 + entropy','55.8%','44.2%','93.1%','81.8%'),
('Selected: ROI 0.70 / GenAI 0.50 + entropy','52.9%','47.1%','94.5%','86.4%')]

LIMIT_ROWS=[
('Single-center, small cohort','High variance and site-specific cues','External multicenter and device-stratified validation'),
('Only seven CSOM test images','Wide minority-sensitive intervals','Targeted chronic-disease recruitment and adjudication'),
('37 labels pending ENT consensus','Potential label error in training additions','Two-reviewer independent labels plus adjudication'),
('JPG-ASOM association','Residual acquisition shortcut risk','Balanced device/file formats and counterfactual audits'),
('Image-level split without patient IDs','Unconfirmed repeated-patient separation','Patient identifiers and grouped splitting'),
('64 x 64 cVAE synthesis','Limited fine-anatomy fidelity','Higher-resolution diffusion with memorization audit'),
('No prospective workflow study','Unknown clinical workload and reader interaction','DECIDE-AI reader and referral study')]

# ---------- Builder functions ----------
def add_front(doc, checkpoint=None):
    p=doc.add_paragraph(style='Title'); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.add_run(TITLE)
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    r=p.add_run('Ankit Kumar Singh1,*; [co-authors to be inserted]2'); r.bold=True; r.font.size=Pt(10.5)
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after=Pt(3)
    p.add_run('1 Department of Electronics and Communication Engineering, National Institute of Technology Raipur, Raipur, India\n')
    p.add_run('2 Department of Otorhinolaryngology, All India Institute of Medical Sciences Raipur, Raipur, India\n')
    p.add_run('* Corresponding author: [institutional e-mail to be inserted]')
    if checkpoint:
        p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        r=p.add_run(checkpoint); r.bold=True; r.font.name='Arial'; r.font.size=Pt(10.5); r.font.color.rgb=RGBColor(35,87,137)
    doc.add_paragraph('Abstract',style='Heading 1'); doc.add_paragraph(ABSTRACT,style='Abstract ESWA')
    p=doc.add_paragraph(); r=p.add_run('Keywords: '); r.bold=True; p.add_run('Generative AI; otoscopy; data provenance; conditional VAE; uncertainty; referral triage')
    doc.add_paragraph('Highlights',style='Heading 1'); add_bullets(doc,HIGHLIGHTS)


def add_intro_related(doc, include_refs=True):
    doc.add_paragraph('1. Introduction',style='Heading 1')
    for p in INTRO_PARAS: doc.add_paragraph(p)
    doc.add_paragraph('1.1 Study objectives and contributions',style='Heading 2'); add_bullets(doc,CONTRIBS)
    doc.add_paragraph('1.2 Research questions',style='Heading 2')
    rqs=[
        'RQ1: Does duplicate- and provenance-aware cohort governance alter the evidence available for model development?',
        'RQ2: Does label-independent conditional encoding provide a useful generative evidence view without true-label leakage?',
        'RQ3: Do generated minority candidates satisfy a null-eligible expert-validation utility gate?',
        'RQ4: Does ROI evidence fusion yield stronger macro discrimination than the latent GenAI ensemble on the same expert test images?',
        'RQ5: Can cross-view agreement and uncertainty concentrate errors in a clinically reviewable queue?']
    add_bullets(doc,rqs)
    doc.add_paragraph('2. Related work',style='Heading 1')
    for h,t in RELATED_PARAS: doc.add_paragraph(h,style='Heading 2'); doc.add_paragraph(t)
    add_table(doc,['Study','Year','Data scale','Task','Method','Reported result','Comparability note'],PUBLISHED_ROWS,[1.05,.45,.85,1.0,1.0,1.15,1.3],7.0,'Table 1. Representative otoscopic AI studies and the current evidence boundary.')
    doc.add_paragraph('2.6 Gap addressed by the present system',style='Heading 2')
    doc.add_paragraph("The literature contains capable image classifiers, generative models, and uncertainty methods, but fewer studies combine data provenance, null-eligible generated-evidence admission, complementary evidence views, probability calibration, and a predefined human-referral action. PG-GECR addresses this integration gap and reports negative generative evidence as part of the system result.")
    if include_refs:
        doc.add_paragraph('References',style='Heading 1')
        for i,r in enumerate(REFS,1): doc.add_paragraph(f'[{i}] {r}',style='Reference ESWA')


def add_methods_arch(doc):
    doc.add_paragraph('3. Materials and dataset governance',style='Heading 1')
    for h,t in METHODS_SECTIONS: doc.add_paragraph(h,style='Heading 2'); doc.add_paragraph(t)
    add_figure(doc,FIG1,'Fig. 1. Provenance-aware construction of the QA-724 research set. The evaluation firewall keeps pseudo-labelled and generated images outside the expert-only validation and test partitions.',6.65)
    add_table(doc,['Partition','AOM','ASOM','CSOM','Normal','Total','Reference status'],SPLIT_ROWS,[1.15,.55,.55,.55,.65,.55,1.7],7.8,'Table 2. Fixed class-stratified data partition and reference-status firewall.')
    add_figure(doc,FIG2,'Fig. 2. Representative field-cropped expert-labelled otoscopic images. Variable illumination, overlapping morphology, and chronic structural heterogeneity motivate complementary evidence views and referral.',6.45)
    doc.add_paragraph('4. Proposed PG-GECR GenAI expert system',style='Heading 1')
    doc.add_paragraph("PG-GECR is a modular intelligent system rather than a renamed classifier. Data-governance rules constrain the evidence available to representation learning; the generative and ROI streams produce independent probabilities; validation controls synthetic admission and branch weights; calibration controls probability sharpness; and the referral controller converts disagreement or uncertainty into a human action.")
    add_figure(doc,FIG3,'Fig. 3. Three-lane PG-GECR architecture. Solid arrows denote inference or governed training streams; dashed feedback loops denote generated-candidate rejection and ENT adjudication.',6.65)
    for h,t in ARCH_SECTIONS:
        doc.add_paragraph(h,style='Heading 2'); doc.add_paragraph(t)
        if h.startswith('4.1'): add_table(doc,['Rule','Module','Operational statement'],RULE_ROWS,[.55,1.5,4.8],8.2,'Table 3. Explicit expert-system knowledge rules.')
        if h.startswith('4.2'):
            add_equation(doc,'q_phi(z | x,y) = N(z; mu_phi(x,y), diag(sigma_phi^2(x,y)))',1)
            add_equation(doc,'z = mu + sigma * epsilon,   epsilon ~ N(0,I)',2)
            add_equation(doc,'L_cVAE = E_q[||x - x_hat||_2^2] + beta D_KL(q_phi(z|x,y) || N(0,I))',3)
        if h.startswith('4.3'): add_equation(doc,'h(x) = [mu_phi(x,AOM) || mu_phi(x,ASOM) || mu_phi(x,CSOM) || mu_phi(x,Normal)] in R^192',4)
        if h.startswith('4.4'):
            add_equation(doc,'U(S) = Delta F1_macro,val + 0.5 Delta BA_val - 0.25 Delta NLL_val - lambda_d D(S) - lambda_m M(S)',5)
            add_equation(doc,'S* = argmax_{S in C union {empty set}} U(S)',6)
        if h.startswith('4.5'): add_equation(doc,'p_ROI(x) = 0.60 p_expert(x) + 0.40 p_QA(x)',7)
        if h.startswith('4.6'):
            add_equation(doc,'p_k(T) = exp(log p_k / T) / sum_j exp(log p_j / T)',8)
            add_equation(doc,'H_n(p) = -sum_k p_k log p_k / log 4',9)
            add_equation(doc,'Retain(x)=1 iff y_ROI=y_GenAI, max p_ROI>=0.70, max p_GenAI>=0.50, H_ROI<=0.65, H_GenAI<=0.75',10)


def add_experiments(doc):
    doc.add_paragraph('5. Experimental protocol',style='Heading 1')
    for h,t in EXP_SECTIONS: doc.add_paragraph(h,style='Heading 2'); doc.add_paragraph(t)
    add_table(doc,['Component','Final setting'],HYPER_ROWS,[2.0,4.8],8.0,'Table 4. Reproducible hyperparameters and final operating settings.')
    doc.add_paragraph('5.6 Executable algorithm',style='Heading 2')
    code=[
        'Algorithm 1: PG-GECR training and referral',
        'Input: expert cohort E, review pool R, fixed validation V and test T',
        '1  Audit hashes, duplicates, file domains, readability, and label conflicts.',
        '2  Split duplicate-clean expert images into training, V, and T.',
        '3  Admit pseudo-label r in R only after unanimous, high-margin, non-duplicate screening.',
        '4  Train cVAE on expert training images; encode every image under four class hypotheses.',
        '5  Generate minority candidates and evaluate each candidate subset against the null set.',
        '6  Reject generated subsets unless expert-only validation utility is larger than the QA baseline.',
        '7  Train expert-ROI and QA-ROI models; select fusion weight on V.',
        '8  Fit temperature parameters on V; lock all thresholds and model versions.',
        '9  For a new image, obtain ROI and latent probabilities, confidence, and entropy.',
        '10 Retain only if branches agree and all confidence/entropy constraints pass; otherwise refer.',
        'Output: calibrated probabilities, class, referral action, provenance and audit record.'
    ]
    for line in code: doc.add_paragraph(line,style='Code ESWA')
    doc.add_paragraph('5.7 Performance formulas',style='Heading 2')
    add_equation(doc,'Balanced accuracy = (1/K) sum_k TP_k / (TP_k + FN_k)',11)
    add_equation(doc,'Macro F1 = (1/K) sum_k 2 Precision_k Recall_k / (Precision_k + Recall_k)',12)
    add_equation(doc,'Coverage(tau) = |{x : Retain_tau(x)=1}| / N',13)
    add_equation(doc,'Error capture(tau) = errors referred at tau / all errors',14)


def add_results_discussion(doc, include_refs=True):
    doc.add_paragraph('6. Results',style='Heading 1')
    for h,t in RESULT_SECTIONS:
        doc.add_paragraph(h,style='Heading 2')
        # Place interpretation before large visual elements so the review copy maintains
        # continuous page flow and does not strand a short paragraph beneath a page break.
        doc.add_paragraph(t)
        if h.startswith('6.2'):
            add_figure(doc,FIG4,'Fig. 4. Conditional generative evidence. (a) cVAE convergence; (b) generated disease candidates; and (c) the validation-utility landscape. The null candidate set remains eligible and was selected.',6.25)
        if h.startswith('6.3'):
            add_figure(doc,FIG5,'Fig. 5. Same-split comparison of candidate classifiers and evidence views. The dual-view ROI probability stream was selected on validation evidence; the latent GenAI branch remained available for consensus referral.',6.25)
            add_table(doc,['Model / evidence view','Accuracy','Balanced acc.','Macro F1','Macro AUC','MCC'],MODEL_ROWS,[2.0,.8,.9,.8,.8,.7],7.6,'Table 5. Expert-only test performance under the fixed split.')
        if h.startswith('6.4'):
            add_table(doc,['Metric','Point difference','Paired bootstrap 95% interval','Interpretation'],DELTA_ROWS,[1.25,1.1,1.8,2.65],8.0,'Table 6. Paired ROI-fusion minus latent-ensemble differences on identical test images.')
        if h.startswith('6.5'):
            add_figure(doc,FIG6,'Fig. 6. Final dual-view ROI evidence performance on the expert-only test set: confusion matrix with row proportions and class-wise ROC curves.',6.25)
            add_table(doc,['Class','Precision','Recall','F1','Support','AUC'],CLASS_ROWS,[1.1,.9,.9,.9,.8,.9],8.2,'Table 7. Class-wise expert-only test results for the final probability stream.')
        if h.startswith('6.6'):
            add_figure(doc,FIG7,'Fig. 7. Probability reliability and secondary disease-versus-Normal decision-curve analysis. Clinical threshold selection requires external calibration.',6.25)
        if h.startswith('6.7'):
            add_figure(doc,FIG8,'Fig. 8. Cross-view consensus referral. Color encodes error capture across risk-coverage operating points; the selected gate retains 52.9% of cases at 94.5% accuracy.',6.25)
            add_table(doc,['Referral rule','Coverage','Referral','Retained accuracy','Error capture'],REFERRAL_ROWS,[2.55,.85,.85,1.05,1.0],8.0,'Table 8. Selected consensus-referral operating points.')
        if h.startswith('6.8'):
            add_figure(doc,FIG9,'Fig. 9. Nonparametric bootstrap analysis. (a) Final metric intervals; (b) paired evidence-view differences. Intervals crossing zero are shown as inconclusive.',6.25)
    doc.add_paragraph('7. Discussion',style='Heading 1')
    heads=['7.1 Principal finding','7.2 Relation to prior evidence','7.3 System novelty and evidence-view complementarity','7.4 Clinical workflow interpretation','7.5 Limitations','7.6 Future evaluation']
    for h,p in zip(heads,DISCUSSION_PARAS): doc.add_paragraph(h,style='Heading 2'); doc.add_paragraph(p)
    add_table(doc,['Limitation','Consequence','Required next action'],LIMIT_ROWS,[1.7,2.2,2.7],8.0,'Table 9. Limitations, consequences, and prospective actions.')
    add_figure(doc,FIG10,'Fig. 10. Proposed deployment sequence with parallel evidence services, consensus control, an ENT queue, model-version logging, and adjudication feedback.',6.55)
    doc.add_paragraph('8. Conclusion',style='Heading 1'); doc.add_paragraph(CONCLUSION)
    doc.add_paragraph('Declarations',style='Heading 1')
    decls=[
        ('Ethics approval and consent to participate','[Insert institutional ethics committee name, approval number, approval date, recruitment period, and consent/assent procedure from the approved protocol.]'),
        ('Consent for publication','Not applicable to identifiable patient material; confirm against the institutional protocol before submission.'),
        ('Funding','[Insert funding source and grant number, or state: This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.]'),
        ('Competing interests','The authors declare no competing interests unless otherwise stated.'),
        ('Data availability','Clinical images are restricted by ethics and institutional governance. De-identified manifests, aggregate metrics, and code may be shared subject to institutional approval.'),
        ('Code availability','The executed Python scripts, fixed-split prediction files, metric tables, and figure-generation workflow accompany the reproducibility package.'),
        ('Author contributions','[Insert CRediT roles for all authors before submission.]'),
        ('Acknowledgements','[Insert acknowledgements, or remove this statement if none.]'),
        ('Declaration of generative AI and AI-assisted technologies in manuscript preparation','During preparation, the authors used OpenAI ChatGPT for manuscript organization, language editing, code drafting, and consistency checks. The authors executed or reviewed the reported analyses, verified the outputs, edited the manuscript, and accept full responsibility for the submitted content.')]
    for h,t in decls:
        p=doc.add_paragraph(); r=p.add_run(h+': '); r.bold=True; p.add_run(t)
    if include_refs:
        doc.add_paragraph('Appendix A. Operational model card',style='Heading 1')
        doc.add_paragraph('PG-GECR is an image-level screening and referral system for research evaluation in tele-otoscopy. Its intended users are trained image-acquisition personnel and ENT reviewers. The system accepts a field-cropped otoscopic image and provenance metadata, and returns calibrated class probabilities, an uncertainty summary, a retained-or-referred action, and a versioned audit record. It does not issue treatment advice, discharge a patient, or override symptoms and clinical examination.')
        add_table(doc,['Model-card field','Locked specification'],[
            ['Primary task','Four-class image screening with human-supervised referral'],
            ['Evaluation boundary','Duplicate-clean, expert-only validation and test partitions'],
            ['Fail-safe behavior','Refer branch disagreement, low confidence, high entropy, or poor image quality'],
            ['Known weak point','Small CSOM support and inadequate stand-alone chronic-disease recall'],
            ['Monitoring requirement','Track device domain, calibration, class mix, referral workload, drift, and overrides']
        ],[1.65,5.15],8.0,'Table A1. Operational model card and claim boundary.')
        doc.add_paragraph('References',style='Heading 1')
        for i,r in enumerate(REFS,1): doc.add_paragraph(f'[{i}] {r}',style='Reference ESWA')


def finalize_doc(doc, outpath):
    # core properties
    doc.core_properties.title=TITLE; doc.core_properties.subject='Expert Systems with Applications submission manuscript'; doc.core_properties.author='Ankit Kumar Singh and co-authors'; doc.core_properties.keywords='Generative AI, otoscopy, expert system, referral triage, QA-724'
    doc.core_properties.comments='Submission draft: complete bracketed author and ethics fields before upload.'
    doc.save(outpath)
    return outpath

# Checkpoint 02
cp2=setup_doc('Expert Systems with Applications - Checkpoint 02')
# The standalone literature checkpoint uses a compact reference style so the bibliography
# closes without a nearly empty terminal page; the full manuscript keeps the larger style.
cp2.styles['Reference ESWA'].font.size=Pt(8.15)
cp2.styles['Reference ESWA'].paragraph_format.line_spacing=1.0
cp2.styles['Reference ESWA'].paragraph_format.space_after=Pt(1.1)
add_front(cp2,'Checkpoint 02: Introduction, critical evidence review, and 60-reference library')
add_intro_related(cp2,include_refs=True)
CP2=finalize_doc(cp2,OUT/'ESWA_PG_GECR_Checkpoint_02_Literature_and_References.docx')

# Checkpoint 03
cp3=setup_doc('Expert Systems with Applications - Checkpoint 03')
add_front(cp3,'Checkpoint 03: Dataset governance, PG-GECR algorithm, and advanced architecture')
add_methods_arch(cp3)
CP3=finalize_doc(cp3,OUT/'ESWA_PG_GECR_Checkpoint_03_Method_and_Architecture.docx')

# Checkpoint 04
cp4=setup_doc('Expert Systems with Applications - Checkpoint 04')
add_front(cp4,'Checkpoint 04: Experimental protocol, hyperparameters, and reproducible implementation')
add_experiments(cp4)
CP4=finalize_doc(cp4,OUT/'ESWA_PG_GECR_Checkpoint_04_Experiments_and_Reproducibility.docx')

# Checkpoint 05
cp5=setup_doc('Expert Systems with Applications - Checkpoint 05')
add_front(cp5,'Checkpoint 05: Results, statistics, clinical interpretation, and limitations')
add_results_discussion(cp5,include_refs=True)
CP5=finalize_doc(cp5,OUT/'ESWA_PG_GECR_Checkpoint_05_Results_and_Discussion.docx')

# Full manuscript / checkpoint 06
final=setup_doc(SHORT)
add_front(final,None)
add_intro_related(final,include_refs=False)
add_methods_arch(final)
add_experiments(final)
add_results_discussion(final,include_refs=True)
FINAL=finalize_doc(final,OUT/'ESWA_PG_GECR_Elsevier_Submission_Ready_Manuscript.docx')

# Separate highlights file
hdoc=setup_doc('Article highlights')
p=hdoc.add_paragraph(style='Title'); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.add_run('Highlights')
p=hdoc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.add_run(TITLE)
add_bullets(hdoc,HIGHLIGHTS)
HFILE=finalize_doc(hdoc,OUT/'ESWA_PG_GECR_Highlights.docx')

# Submission checklist
cdoc=setup_doc('Submission readiness checklist')
p=cdoc.add_paragraph(style='Title'); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.add_run('Expert Systems with Applications submission checklist')
checks=[
('Manuscript','Final single-column editable Word file; continuous flow; editable tables; 60 numbered references.'),
('Highlights','Separate editable file with five bullets; each bullet should be checked in the submission portal for the 85-character limit.'),
('Graphical abstract','Separate 1328 x 531 pixel, 300 dpi PNG; also embedded in the package.'),
('Ethics','Replace bracketed committee, approval number, date, recruitment, consent, and assent fields.'),
('Authorship','Insert all authors, affiliations, correspondence e-mail, CRediT roles, and final author order.'),
('Funding and conflicts','Insert verified funding statement and competing-interest declarations.'),
('Pseudo-label status','Obtain two-ENT independent review and adjudication before presenting the 37 additions as clinical ground truth.'),
('Data and code','Confirm institutionally permitted sharing level and repository or controlled-access statement.'),
('Clinical claims','Retain screening/referral wording; do not describe the system as autonomous diagnosis.'),
('Files','Upload manuscript, highlights, graphical abstract, and any permitted reproducibility supplement as separate files.')]
add_table(cdoc,['Item','Required author action'],checks,[1.5,5.3],9.0)
CHECK=finalize_doc(cdoc,OUT/'ESWA_PG_GECR_Submission_Checklist.docx')

# Copy reproducibility zip and source stats into package
shutil.copy2('/mnt/data/ESWA_GenAI_QA724_Reproducibility_Package.zip',OUT/'ESWA_GenAI_QA724_Reproducibility_Package.zip')
# save metric table CSV
mdf.to_csv(OUT/'same_split_evidence_view_comparison.csv',index=False)

# XML scan for prohibited stem
for fp in [CP2,CP3,CP4,CP5,FINAL,HFILE,CHECK]:
    with zipfile.ZipFile(fp) as z:
        text=' '.join(z.read(n).decode('utf-8','ignore') for n in z.namelist() if n.endswith('.xml'))
    bad=re.findall(r'(?i)improv\w*',text)
    if bad:
        raise RuntimeError(f'Prohibited word stem in {fp.name}: {sorted(set(bad))[:20]}')

# Create package zip after PDF rendering by outer shell; preliminary here
print(json.dumps({
    'checkpoint_02':str(CP2),'checkpoint_03':str(CP3),'checkpoint_04':str(CP4),'checkpoint_05':str(CP5),
    'final_docx':str(FINAL),'highlights':str(HFILE),'graphical_abstract':str(GRAPHICAL),'checklist':str(CHECK),
    'figures':[str(p) for p in sorted(FIGDIR.glob('*.png'))]
},indent=2))
