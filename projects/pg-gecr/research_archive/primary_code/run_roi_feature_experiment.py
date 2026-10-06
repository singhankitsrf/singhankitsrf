from __future__ import annotations
import os, json, math, warnings, random
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd, cv2
from PIL import Image, ImageOps
from scipy.stats import skew, kurtosis
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import *
from sklearn.calibration import CalibratedClassifierCV
from scipy.optimize import minimize_scalar
warnings.filterwarnings('ignore')
SEED=42; random.seed(SEED); np.random.seed(SEED)
ROOT=Path('/mnt/data/eswa_final_work')
DATA=ROOT/'Dataset'; TRAIN=DATA/'Training dataset/Otitis Media'; TEST=DATA/'Testing dataset'
META=Path('/mnt/data/eswa_genai_project/test_ai_pseudo_labels_all_204.csv')
OUT=ROOT/'roi_results'; OUT.mkdir(exist_ok=True)
CLASSES=['AOM','ASOM','CSOM','Normal']; C2I={c:i for i,c in enumerate(CLASSES)}
F2C={'AOM':'AOM','ASOM':'ASOM','CSOM':'CSOM','Normal Tympanic Membrane':'Normal'}
REMOVE={'AOM/aom (11).png','Normal Tympanic Membrane/normal (53).png','AOM/aom (8).png','Normal Tympanic Membrane/normal (51).png','AOM/aom (9).png','Normal Tympanic Membrane/normal (52).png','ASOM/asom (23).JPG','ASOM/asom (28).JPG','ASOM/asom (30).JPG','ASOM/asom (41).JPG','ASOM/asom (75).JPG','ASOM/asom (7).JPG'}

def load_crop(path,size=128):
    a=np.fromfile(str(path),np.uint8); b=cv2.imdecode(a,cv2.IMREAD_COLOR)
    if b is None: raise ValueError(path)
    g=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY); rgb=cv2.cvtColor(b,cv2.COLOR_BGR2RGB)
    m=(g>12).astype(np.uint8)*255; n,l,s,c=cv2.connectedComponentsWithStats(m,8)
    if n>1:
        h,w=g.shape; cand=[]
        for i in range(1,n):
            x,y,ww,hh,area=s[i]; cx,cy=c[i]; pen=((cx-w/2)**2+(cy-h/2)**2)/(h*h+w*w)
            cand.append((area*(1-.6*pen),x,y,ww,hh))
        _,x,y,ww,hh=max(cand); pad=int(.03*max(ww,hh)); x=max(0,x-pad); y=max(0,y-pad); x2=min(w,x+ww+2*pad); y2=min(h,y+hh+2*pad); rgb=rgb[y:y2,x:x2]
    return np.asarray(ImageOps.fit(Image.fromarray(rgb),(size,size),Image.Resampling.LANCZOS),np.uint8)

def moments(a):
    a=a.astype(np.float32).reshape(-1)
    return [a.mean(),a.std(),np.percentile(a,5),np.percentile(a,25),np.median(a),np.percentile(a,75),np.percentile(a,95),skew(a),kurtosis(a)]

def feature_one(path):
    rgb=load_crop(path,128)
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)
    f=[]
    # channel moments and histograms
    for arr in [rgb,hsv,lab]:
        for ch in range(3):
            f.extend(moments(arr[:,:,ch]))
            hist=cv2.calcHist([arr],[ch],None,[16],[0,256]).flatten(); hist=hist/(hist.sum()+1e-9); f.extend(hist)
    # LBP multi-radius
    for P,R in [(8,1),(16,2),(24,3)]:
        lbp=local_binary_pattern(gray,P,R,method='uniform')
        hist,_=np.histogram(lbp.ravel(),bins=np.arange(0,P+3),range=(0,P+2),density=True); f.extend(hist)
    # HOG at two scales
    f.extend(hog(gray,orientations=9,pixels_per_cell=(16,16),cells_per_block=(2,2),block_norm='L2-Hys',feature_vector=True))
    small=cv2.resize(gray,(64,64),interpolation=cv2.INTER_AREA)
    f.extend(hog(small,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),block_norm='L2-Hys',feature_vector=True))
    # GLCM
    q=(gray//16).astype(np.uint8)
    gl=graycomatrix(q,distances=[1,2,4],angles=[0,np.pi/4,np.pi/2,3*np.pi/4],levels=16,symmetric=True,normed=True)
    for prop in ['contrast','dissimilarity','homogeneity','energy','correlation','ASM']:
        f.extend(graycoprops(gl,prop).ravel())
    # shape/edge/frequency/quality
    edges=cv2.Canny(gray,40,120)
    lap=cv2.Laplacian(gray,cv2.CV_64F)
    gx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3); gy=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
    mag=np.sqrt(gx*gx+gy*gy)
    fft=np.fft.fftshift(np.fft.fft2(gray)); spec=np.log1p(np.abs(fft)); h,w=gray.shape; cy,cx=h//2,w//2
    yy,xx=np.ogrid[:h,:w]; r=np.sqrt((yy-cy)**2+(xx-cx)**2)
    f.extend([edges.mean()/255,lap.var(),mag.mean(),mag.std(),shannon_entropy(gray),gray.mean(),gray.std(),
              np.mean(hsv[:,:,1])/255,np.mean(hsv[:,:,2])/255,
              spec[r<8].mean(),spec[(r>=8)&(r<24)].mean(),spec[r>=24].mean()])
    # low-res thumbnail captures global morphology
    thumb=cv2.resize(rgb,(16,16),interpolation=cv2.INTER_AREA).astype(np.float32)/255
    f.extend(thumb.ravel())
    return np.asarray(f,dtype=np.float32)

expert=[]
for folder,c in F2C.items():
    for p in sorted((TRAIN/folder).glob('*')):
        rel=f'{folder}/{p.name}'
        if p.is_file() and rel not in REMOVE: expert.append({'path':p,'label':c,'source':'expert','rel':rel})
assert len(expert)==687
idx=np.arange(len(expert)); y=np.array([C2I[r['label']] for r in expert])
tr,tmp=train_test_split(idx,test_size=.30,stratify=y,random_state=SEED); va,te=train_test_split(tmp,test_size=.5,stratify=y[tmp],random_state=SEED)
trr=[expert[i] for i in tr]; var=[expert[i] for i in va]; ter=[expert[i] for i in te]
pdf=pd.read_csv(META); pseudo=[]
for _,r in pdf[pdf.included_in_QA_clean_724==True].iterrows(): pseudo.append({'path':TEST/r.original_filename,'label':r.ai_predicted_label,'source':'pseudo'})
assert len(pseudo)==37
records=trr+var+ter+pseudo
cache=OUT/'roi_features.npz'
if cache.exists():
    z=np.load(cache); X=z['X']; Y=z['Y']
else:
    feats=[]
    for i,r in enumerate(records):
        feats.append(feature_one(r['path']))
        if (i+1)%100==0: print('features',i+1,flush=True)
    X=np.stack(feats); Y=np.array([C2I[r['label']] for r in records]); np.savez_compressed(cache,X=X,Y=Y)
print('shape',X.shape,Counter(Y),flush=True)
a=len(trr); b=a+len(var); c=b+len(ter)
Xtr,Xv,Xt,Xp=X[:a],X[a:b],X[b:c],X[c:]; ytr,yv,yt,yp=Y[:a],Y[a:b],Y[b:c],Y[c:]

def met(y,p):
    pr=p.argmax(1); pre,rec,f1,_=precision_recall_fscore_support(y,pr,average='macro',zero_division=0)
    auc=roc_auc_score(label_binarize(y,classes=np.arange(4)),p,multi_class='ovr',average='macro')
    return {'accuracy':accuracy_score(y,pr),'balanced_accuracy':balanced_accuracy_score(y,pr),'macro_precision':pre,'macro_recall':rec,'macro_f1':f1,'macro_auc':auc,'MCC':matthews_corrcoef(y,pr),'log_loss':log_loss(y,p,labels=np.arange(4))}

def eval_model(name, model, XX, yy):
    model.fit(XX,yy); pv=model.predict_proba(Xv); pt=model.predict_proba(Xt)
    return {'name':name,'model':model,'pval':pv,'ptest':pt,'val':met(yv,pv),'test':met(yt,pt)}

# Diverse but bounded grid
configs=[]
for C in [.03,.1,.3,1,3,10]:
    configs.append(('ROI Logistic',Pipeline([('s',StandardScaler()),('p',PCA(n_components=.97,whiten=True,random_state=SEED)),('m',LogisticRegression(C=C,max_iter=5000,class_weight='balanced',random_state=SEED))])))
for C in [.3,1,3,10,30]:
    configs.append(('ROI RBF-SVM',Pipeline([('s',StandardScaler()),('p',PCA(n_components=.97,whiten=True,random_state=SEED)),('m',SVC(C=C,gamma='scale',probability=True,class_weight='balanced',random_state=SEED))])))
for n,mf,leaf in [(500,'sqrt',1),(700,.4,1),(700,.6,2),(900,.8,2)]:
    configs.append(('ROI Extra Trees',ExtraTreesClassifier(n_estimators=n,max_features=mf,min_samples_leaf=leaf,class_weight='balanced',random_state=SEED,n_jobs=-1)))
for n,mf,leaf in [(500,'sqrt',1),(700,.4,1),(700,.6,2),(900,.8,2)]:
    configs.append(('ROI Random Forest',RandomForestClassifier(n_estimators=n,max_features=mf,min_samples_leaf=leaf,class_weight='balanced_subsample',random_state=SEED,n_jobs=-1)))
for hid,alpha in [((128,),1e-3),((128,64),1e-3),((256,96),3e-4)]:
    configs.append(('ROI MLP',Pipeline([('s',StandardScaler()),('p',PCA(n_components=.98,whiten=True,random_state=SEED)),('m',MLPClassifier(hidden_layer_sizes=hid,alpha=alpha,early_stopping=True,validation_fraction=.15,max_iter=1000,random_state=SEED))])))
try:
    from lightgbm import LGBMClassifier
    for leaves,depth in [(15,5),(31,7),(15,-1)]:
        configs.append(('ROI LightGBM',LGBMClassifier(n_estimators=350,learning_rate=.025,num_leaves=leaves,max_depth=depth,subsample=.85,colsample_bytree=.7,class_weight='balanced',reg_lambda=2,random_state=SEED,verbosity=-1,n_jobs=-1)))
except Exception as e: print('no lightgbm',e)

regimes={'Expert-only':(Xtr,ytr),'QA-724':(np.vstack([Xtr,Xp]),np.r_[ytr,yp])}
rows=[]; bests={}
for reg,(XX,yy) in regimes.items():
    fam={}
    for i,(name,model) in enumerate(configs):
        r=eval_model(name,model,XX,yy)
        if name not in fam or r['val']['macro_f1']>fam[name]['val']['macro_f1']: fam[name]=r
        print(reg,name,i,round(r['val']['macro_f1'],3),round(r['test']['macro_f1'],3),flush=True)
    for name,r in fam.items(): rows.append({'regime':reg,'model':name,**{'val_'+k:v for k,v in r['val'].items()},**{'test_'+k:v for k,v in r['test'].items()}})
    best=max(fam.values(),key=lambda r:(r['val']['macro_f1'],r['val']['balanced_accuracy'],r['val']['macro_auc']))
    bests[reg]=best
    print('BEST',reg,best['name'],best['val'],best['test'],flush=True)

# validation weighted ensemble among top distinct families in QA regime
XX,yy=regimes['QA-724']; fam={}
for name,model in configs:
    r=eval_model(name,model,XX,yy)
    if name not in fam or r['val']['macro_f1']>fam[name]['val']['macro_f1']: fam[name]=r
selected=sorted(fam.values(),key=lambda r:r['val']['macro_f1'],reverse=True)[:4]
w=np.array([r['val']['macro_f1']**3 for r in selected]); w/=w.sum()
pv=sum(wi*r['pval'] for wi,r in zip(w,selected)); pt=sum(wi*r['ptest'] for wi,r in zip(w,selected))
ens={'name':'ROI validation-weighted ensemble','pval':pv,'ptest':pt,'val':met(yv,pv),'test':met(yt,pt),'weights':dict(zip([r['name'] for r in selected],w.tolist()))}
print('ENSEMBLE',ens['weights'],ens['val'],ens['test'])
# Temperature scaling
def ts(p,T):
 z=np.log(np.clip(p,1e-9,1))/T; z-=z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)
opt=minimize_scalar(lambda T:log_loss(yv,ts(pv,T),labels=np.arange(4)),bounds=(.25,4),method='bounded')
T=float(opt.x); pct=ts(pt,T); final=met(yt,pct)
print('T',T,'ROI FINAL',final)
rows.append({'regime':'QA-724','model':'ROI validation-weighted ensemble (calibrated)',**{'val_'+k:v for k,v in ens['val'].items()},**{'test_'+k:v for k,v in final.items()}})
pd.DataFrame(rows).to_csv(OUT/'roi_model_comparison.csv',index=False)
np.savez_compressed(OUT/'roi_final_predictions.npz',y=yt,probs=pct,pval=pv,yval=yv,temperature=T)
json.dump({'temperature':T,'metrics':final,'validation':ens['val'],'weights':ens['weights'],'best_expert':{'name':bests['Expert-only']['name'],'val':bests['Expert-only']['val'],'test':bests['Expert-only']['test']},'best_QA':{'name':bests['QA-724']['name'],'val':bests['QA-724']['val'],'test':bests['QA-724']['test']}},open(OUT/'roi_summary.json','w'),indent=2)
print('DONE')
