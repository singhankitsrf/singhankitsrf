from __future__ import annotations
import os,random,warnings,json,math,time
from pathlib import Path
from collections import Counter
import numpy as np,pandas as pd,cv2,torch,torch.nn as nn,torch.nn.functional as F
from PIL import Image,ImageOps
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,label_binarize
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import *
from scipy.optimize import minimize_scalar
warnings.filterwarnings('ignore')
SEED=42;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(8)
ROOT=Path('/mnt/data/eswa_final_work');DATA=ROOT/'Dataset';TRAIN=DATA/'Training dataset/Otitis Media';TEST=DATA/'Testing dataset';META=Path('/mnt/data/eswa_genai_project/test_ai_pseudo_labels_all_204.csv');OUT=ROOT/'genai_retrain';OUT.mkdir(exist_ok=True)
CLASSES=['AOM','ASOM','CSOM','Normal'];C2I={c:i for i,c in enumerate(CLASSES)};F2C={'AOM':'AOM','ASOM':'ASOM','CSOM':'CSOM','Normal Tympanic Membrane':'Normal'}
REMOVE={'AOM/aom (11).png','Normal Tympanic Membrane/normal (53).png','AOM/aom (8).png','Normal Tympanic Membrane/normal (51).png','AOM/aom (9).png','Normal Tympanic Membrane/normal (52).png','ASOM/asom (23).JPG','ASOM/asom (28).JPG','ASOM/asom (30).JPG','ASOM/asom (41).JPG','ASOM/asom (75).JPG','ASOM/asom (7).JPG'}
def crop(path,size=64):
 a=np.fromfile(str(path),np.uint8);b=cv2.imdecode(a,cv2.IMREAD_COLOR);g=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY);rgb=cv2.cvtColor(b,cv2.COLOR_BGR2RGB);m=(g>12).astype(np.uint8)*255;n,l,s,c=cv2.connectedComponentsWithStats(m,8)
 if n>1:
  h,w=g.shape;ss=[]
  for i in range(1,n):
   x,y,ww,hh,area=s[i];cx,cy=c[i];pen=((cx-w/2)**2+(cy-h/2)**2)/(h*h+w*w);ss.append((area*(1-.6*pen),x,y,ww,hh))
  _,x,y,ww,hh=max(ss);pad=int(.03*max(ww,hh));x=max(0,x-pad);y=max(0,y-pad);x2=min(w,x+ww+2*pad);y2=min(h,y+hh+2*pad);rgb=rgb[y:y2,x:x2]
 return np.asarray(ImageOps.fit(Image.fromarray(rgb),(size,size),Image.Resampling.LANCZOS),np.uint8)
expert=[]
for f,c in F2C.items():
 for p in sorted((TRAIN/f).glob('*')):
  if p.is_file() and f'{f}/{p.name}' not in REMOVE:expert.append({'path':p,'label':c})
idx=np.arange(len(expert));y=np.array([C2I[r['label']] for r in expert]);tr,tmp=train_test_split(idx,test_size=.30,stratify=y,random_state=SEED);va,te=train_test_split(tmp,test_size=.5,stratify=y[tmp],random_state=SEED);trr=[expert[i] for i in tr];var=[expert[i] for i in va];ter=[expert[i] for i in te]
pdf=pd.read_csv(META);pseudo=[{'path':TEST/r.original_filename,'label':r.ai_predicted_label} for _,r in pdf[pdf.included_in_QA_clean_724==True].iterrows()]
allr=trr+var+ter+pseudo;arr=np.stack([crop(r['path']) for r in allr]);Ximg=torch.from_numpy(arr.transpose(0,3,1,2)).float()/255.;Y=np.array([C2I[r['label']] for r in allr]);a=len(trr);b=a+len(var);c=b+len(ter);ytr,yv,yt,yp=Y[:a],Y[a:b],Y[b:c],Y[c:]
class CVAE(nn.Module):
 def __init__(self,latent=48,nc=4):
  super().__init__();self.nc=nc;self.enc=nn.Sequential(nn.Conv2d(3+nc,24,4,2,1),nn.BatchNorm2d(24),nn.LeakyReLU(.2),nn.Conv2d(24,48,4,2,1),nn.BatchNorm2d(48),nn.LeakyReLU(.2),nn.Conv2d(48,96,4,2,1),nn.BatchNorm2d(96),nn.LeakyReLU(.2),nn.Conv2d(96,128,4,2,1),nn.BatchNorm2d(128),nn.LeakyReLU(.2));self.mu=nn.Linear(128*4*4,48);self.lv=nn.Linear(128*4*4,48);self.fd=nn.Linear(52,128*4*4);self.dec=nn.Sequential(nn.ConvTranspose2d(128,96,4,2,1),nn.BatchNorm2d(96),nn.ReLU(),nn.ConvTranspose2d(96,48,4,2,1),nn.BatchNorm2d(48),nn.ReLU(),nn.ConvTranspose2d(48,24,4,2,1),nn.BatchNorm2d(24),nn.ReLU(),nn.ConvTranspose2d(24,3,4,2,1),nn.Sigmoid())
 def encode(self,x,y):
  oh=F.one_hot(y,self.nc).float()[:,:,None,None].expand(-1,-1,64,64);h=self.enc(torch.cat([x,oh],1)).flatten(1);return self.mu(h),self.lv(h)
 def decode(self,z,y):
  oh=F.one_hot(y,self.nc).float();return self.dec(self.fd(torch.cat([z,oh],1)).view(-1,128,4,4))
 def forward(self,x,y):
  mu,lv=self.encode(x,y);z=mu+torch.randn_like(mu)*torch.exp(.5*lv);return self.decode(z,y),mu,lv
vae=CVAE();opt=torch.optim.AdamW(vae.parameters(),lr=1.5e-3,weight_decay=1e-5);bs=32;inds=np.arange(a);hist=[]
for ep in range(1,19):
 np.random.shuffle(inds);vae.train();losses=[]
 for st in range(0,a,bs):
  ix=inds[st:st+bs];x=Ximg[ix];yb=torch.from_numpy(ytr[ix]).long();out,mu,lv=vae(x,yb);rec=F.mse_loss(out,x);kl=-.5*torch.mean(1+lv-mu.pow(2)-lv.exp());beta=min(.003,.0005+ep*.00015);loss=rec+beta*kl;opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(vae.parameters(),5);opt.step();losses.append((loss.item(),rec.item(),kl.item()))
 hist.append({'epoch':ep,'loss':np.mean([q[0] for q in losses]),'reconstruction':np.mean([q[1] for q in losses]),'kl':np.mean([q[2] for q in losses])});print('ep',ep,hist[-1],flush=True)
pd.DataFrame(hist).to_csv(OUT/'cvae_history.csv',index=False);torch.save(vae.state_dict(),OUT/'cvae.pt');vae.eval()
def hypfeat(x):
 out=[]
 with torch.no_grad():
  for st in range(0,len(x),64):
   xb=x[st:st+64];parts=[]
   for k in range(4):parts.append(vae.encode(xb,torch.full((len(xb),),k,dtype=torch.long))[0])
   out.append(torch.cat(parts,1).numpy())
 return np.vstack(out)
F_all=hypfeat(Ximg);Xtr,Xv,Xt,Xp=F_all[:a],F_all[a:b],F_all[b:c],F_all[c:]
def met(y,p):
 pred=p.argmax(1);pre,rec,f1,_=precision_recall_fscore_support(y,pred,average='macro',zero_division=0);auc=roc_auc_score(label_binarize(y,classes=np.arange(4)),p,multi_class='ovr',average='macro');return {'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred),'macro_precision':pre,'macro_recall':rec,'macro_f1':f1,'macro_auc':auc,'MCC':matthews_corrcoef(y,pred),'log_loss':log_loss(y,p,labels=np.arange(4))}
XX=np.vstack([Xtr,Xp]);yy=np.r_[ytr,yp]
models=[]
for name,m in [('Latent Logistic',Pipeline([('s',StandardScaler()),('m',LogisticRegression(C=.1,max_iter=2500,class_weight='balanced',random_state=SEED))])),('Latent SVM',Pipeline([('s',StandardScaler()),('m',SVC(C=10,probability=True,class_weight='balanced',random_state=SEED))])),('Latent ExtraTrees',ExtraTreesClassifier(n_estimators=500,max_features=.7,class_weight='balanced',min_samples_leaf=2,random_state=SEED,n_jobs=-1)),('Latent MLP',Pipeline([('s',StandardScaler()),('m',MLPClassifier(hidden_layer_sizes=(96,48),alpha=1e-3,early_stopping=True,max_iter=600,random_state=SEED))]))]:
 m.fit(XX,yy);pv=m.predict_proba(Xv);pt=m.predict_proba(Xt);models.append((name,pv,pt,met(yv,pv),met(yt,pt)));print(name,models[-1][3],models[-1][4])
w=np.array([m[3]['macro_f1'] for m in models]);w/=w.sum();pv=sum(wi*m[1] for wi,m in zip(w,models));pt=sum(wi*m[2] for wi,m in zip(w,models));val=met(yv,pv);unc=met(yt,pt)
def ts(p,T):z=np.log(np.clip(p,1e-9,1))/T;z-=z.max(1,keepdims=True);e=np.exp(z);return e/e.sum(1,keepdims=True)
T=float(minimize_scalar(lambda T:log_loss(yv,ts(pv,T),labels=np.arange(4)),bounds=(.25,4),method='bounded').x);pvc=ts(pv,T);ptc=ts(pt,T);final=met(yt,ptc);print('ENSEMBLE',w,val,final)
np.savez_compressed(OUT/'genai_predictions.npz',y=yt,probs=ptc,yval=yv,pval=pvc,temperature=T);json.dump({'weights':dict(zip([m[0] for m in models],w.tolist())),'validation':val,'test':final,'temperature':T,'members':{m[0]:{'validation':m[3],'test':m[4]} for m in models}},open(OUT/'summary.json','w'),indent=2)
