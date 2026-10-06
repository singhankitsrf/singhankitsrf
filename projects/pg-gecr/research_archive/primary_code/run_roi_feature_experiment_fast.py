from __future__ import annotations
import os, json, warnings, random
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd, cv2
from PIL import Image, ImageOps
from skimage.feature import hog, local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import *
from scipy.optimize import minimize_scalar
from joblib import Parallel, delayed
warnings.filterwarnings('ignore')
SEED=42; random.seed(SEED); np.random.seed(SEED)
ROOT=Path('/mnt/data/eswa_final_work'); DATA=ROOT/'Dataset'; TRAIN=DATA/'Training dataset/Otitis Media'; TEST=DATA/'Testing dataset'; META=Path('/mnt/data/eswa_genai_project/test_ai_pseudo_labels_all_204.csv'); OUT=ROOT/'roi_results_fast'; OUT.mkdir(exist_ok=True)
CLASSES=['AOM','ASOM','CSOM','Normal']; C2I={c:i for i,c in enumerate(CLASSES)}; F2C={'AOM':'AOM','ASOM':'ASOM','CSOM':'CSOM','Normal Tympanic Membrane':'Normal'}
REMOVE={'AOM/aom (11).png','Normal Tympanic Membrane/normal (53).png','AOM/aom (8).png','Normal Tympanic Membrane/normal (51).png','AOM/aom (9).png','Normal Tympanic Membrane/normal (52).png','ASOM/asom (23).JPG','ASOM/asom (28).JPG','ASOM/asom (30).JPG','ASOM/asom (41).JPG','ASOM/asom (75).JPG','ASOM/asom (7).JPG'}

def load_crop(path,size=96):
    a=np.fromfile(str(path),np.uint8); b=cv2.imdecode(a,cv2.IMREAD_COLOR)
    if b is None: raise ValueError(path)
    g=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY); rgb=cv2.cvtColor(b,cv2.COLOR_BGR2RGB)
    m=(g>12).astype(np.uint8)*255; n,l,s,c=cv2.connectedComponentsWithStats(m,8)
    if n>1:
        h,w=g.shape; scores=[]
        for i in range(1,n):
            x,y,ww,hh,area=s[i]; cx,cy=c[i]; pen=((cx-w/2)**2+(cy-h/2)**2)/(h*h+w*w); scores.append((area*(1-.6*pen),x,y,ww,hh))
        _,x,y,ww,hh=max(scores); pad=int(.03*max(ww,hh)); x=max(0,x-pad); y=max(0,y-pad); x2=min(w,x+ww+2*pad); y2=min(h,y+hh+2*pad); rgb=rgb[y:y2,x:x2]
    return np.asarray(ImageOps.fit(Image.fromarray(rgb),(size,size),Image.Resampling.LANCZOS),np.uint8)

def feature_one(path):
    rgb=load_crop(path,96); gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY); hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV); lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)
    f=[]
    for arr in [rgb,hsv,lab]:
        for ch in range(3):
            v=arr[:,:,ch].astype(np.float32)
            f += [float(v.mean()),float(v.std()),float(np.percentile(v,10)),float(np.percentile(v,50)),float(np.percentile(v,90))]
            h=cv2.calcHist([arr],[ch],None,[12],[0,256]).flatten(); h/=h.sum()+1e-9; f.extend(h.tolist())
    for P,R in [(8,1),(16,2)]:
        lbp=local_binary_pattern(gray,P,R,method='uniform'); h,_=np.histogram(lbp.ravel(),bins=np.arange(P+3),range=(0,P+2),density=True); f.extend(h.tolist())
    f.extend(hog(gray,orientations=9,pixels_per_cell=(12,12),cells_per_block=(2,2),block_norm='L2-Hys',feature_vector=True).tolist())
    edges=cv2.Canny(gray,40,120); lap=cv2.Laplacian(gray,cv2.CV_64F); gx=cv2.Sobel(gray,cv2.CV_64F,1,0,3); gy=cv2.Sobel(gray,cv2.CV_64F,0,1,3); mag=np.sqrt(gx*gx+gy*gy)
    f += [float(edges.mean()/255),float(lap.var()),float(mag.mean()),float(mag.std())]
    thumb=cv2.resize(rgb,(12,12),interpolation=cv2.INTER_AREA).astype(np.float32).ravel()/255; f.extend(thumb.tolist())
    return np.asarray(f,np.float32)

expert=[]
for folder,c in F2C.items():
    for p in sorted((TRAIN/folder).glob('*')):
        rel=f'{folder}/{p.name}'
        if p.is_file() and rel not in REMOVE: expert.append({'path':p,'label':c,'source':'expert'})
assert len(expert)==687
idx=np.arange(len(expert)); y=np.array([C2I[r['label']] for r in expert]); tr,tmp=train_test_split(idx,test_size=.30,stratify=y,random_state=SEED); va,te=train_test_split(tmp,test_size=.5,stratify=y[tmp],random_state=SEED)
trr=[expert[i] for i in tr]; var=[expert[i] for i in va]; ter=[expert[i] for i in te]
pdf=pd.read_csv(META); pseudo=[{'path':TEST/r.original_filename,'label':r.ai_predicted_label,'source':'pseudo'} for _,r in pdf[pdf.included_in_QA_clean_724==True].iterrows()]; assert len(pseudo)==37
records=trr+var+ter+pseudo
cache=OUT/'features.npz'
if cache.exists(): z=np.load(cache); X=z['X']; Y=z['Y']
else:
    X=np.stack(Parallel(n_jobs=8,backend='loky',verbose=5)(delayed(feature_one)(r['path']) for r in records)); Y=np.array([C2I[r['label']] for r in records]); np.savez_compressed(cache,X=X,Y=Y)
print('shape',X.shape,flush=True)
a=len(trr);b=a+len(var);c=b+len(ter);Xtr,Xv,Xt,Xp=X[:a],X[a:b],X[b:c],X[c:];ytr,yv,yt,yp=Y[:a],Y[a:b],Y[b:c],Y[c:]

def met(y,p):
    pred=p.argmax(1); pre,rec,f1,_=precision_recall_fscore_support(y,pred,average='macro',zero_division=0); auc=roc_auc_score(label_binarize(y,classes=np.arange(4)),p,multi_class='ovr',average='macro')
    return {'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred),'macro_precision':pre,'macro_recall':rec,'macro_f1':f1,'macro_auc':auc,'MCC':matthews_corrcoef(y,pred),'log_loss':log_loss(y,p,labels=np.arange(4))}

def run(name,m,XX,yy): m.fit(XX,yy); pv=m.predict_proba(Xv);pt=m.predict_proba(Xt);return {'name':name,'model':m,'pval':pv,'ptest':pt,'val':met(yv,pv),'test':met(yt,pt)}
configs=[]
for C in [.05,.2,1,5]: configs.append(('ROI Logistic',Pipeline([('s',StandardScaler()),('p',PCA(n_components=.97,whiten=True,random_state=SEED)),('m',LogisticRegression(C=C,max_iter=4000,class_weight='balanced',random_state=SEED))])))
for C in [.5,2,10]: configs.append(('ROI RBF-SVM',Pipeline([('s',StandardScaler()),('p',PCA(n_components=.97,whiten=True,random_state=SEED)),('m',SVC(C=C,probability=True,class_weight='balanced',random_state=SEED))])))
for n,mf,leaf in [(500,'sqrt',1),(700,.5,1),(700,.7,2)]: configs.append(('ROI Extra Trees',ExtraTreesClassifier(n_estimators=n,max_features=mf,min_samples_leaf=leaf,class_weight='balanced',random_state=SEED,n_jobs=-1)))
for n,mf,leaf in [(500,'sqrt',1),(700,.5,1),(700,.7,2)]: configs.append(('ROI Random Forest',RandomForestClassifier(n_estimators=n,max_features=mf,min_samples_leaf=leaf,class_weight='balanced_subsample',random_state=SEED,n_jobs=-1)))
for hid,alpha in [((96,),1e-3),((128,64),5e-4)]: configs.append(('ROI MLP',Pipeline([('s',StandardScaler()),('p',PCA(n_components=.98,whiten=True,random_state=SEED)),('m',MLPClassifier(hidden_layer_sizes=hid,alpha=alpha,early_stopping=True,max_iter=700,random_state=SEED))])))
try:
 from lightgbm import LGBMClassifier
 for leaves in [15,31]: configs.append(('ROI LightGBM',LGBMClassifier(n_estimators=300,learning_rate=.03,num_leaves=leaves,max_depth=7,subsample=.85,colsample_bytree=.75,class_weight='balanced',reg_lambda=2,random_state=SEED,verbosity=-1,n_jobs=-1)))
except: pass
rows=[]; best_fam={}; saved={}
for reg,XX,yy in [('Expert-only',Xtr,ytr),('QA-724',np.vstack([Xtr,Xp]),np.r_[ytr,yp])]:
 fam={}
 for name,m in configs:
  r=run(name,m,XX,yy)
  if name not in fam or r['val']['macro_f1']>fam[name]['val']['macro_f1']: fam[name]=r
  print(reg,name,round(r['val']['macro_f1'],3),round(r['test']['macro_f1'],3),flush=True)
 for name,r in fam.items(): rows.append({'regime':reg,'model':name,**{'val_'+k:v for k,v in r['val'].items()},**{'test_'+k:v for k,v in r['test'].items()}})
 saved[reg]=fam
# validation ensemble on QA
fam=saved['QA-724']; selected=sorted(fam.values(),key=lambda r:r['val']['macro_f1'],reverse=True)[:4]; w=np.array([r['val']['macro_f1']**3 for r in selected]);w/=w.sum();pv=sum(wi*r['pval'] for wi,r in zip(w,selected));pt=sum(wi*r['ptest'] for wi,r in zip(w,selected)); val=met(yv,pv);test=met(yt,pt)
def ts(p,T):
 z=np.log(np.clip(p,1e-9,1))/T;z-=z.max(1,keepdims=True);e=np.exp(z);return e/e.sum(1,keepdims=True)
T=float(minimize_scalar(lambda T:log_loss(yv,ts(pv,T),labels=np.arange(4)),bounds=(.25,4),method='bounded').x); pct=ts(pt,T);final=met(yt,pct)
rows.append({'regime':'QA-724','model':'ROI validation ensemble (calibrated)',**{'val_'+k:v for k,v in val.items()},**{'test_'+k:v for k,v in final.items()}})
pd.DataFrame(rows).to_csv(OUT/'model_comparison.csv',index=False);np.savez_compressed(OUT/'final_predictions.npz',y=yt,probs=pct,yval=yv,pval=pv,temperature=T)
json.dump({'weights':dict(zip([r['name'] for r in selected],w.tolist())),'validation':val,'test':final,'temperature':T},open(OUT/'summary.json','w'),indent=2)
print('FINAL',dict(zip([r['name'] for r in selected],w)),val,final,flush=True)
