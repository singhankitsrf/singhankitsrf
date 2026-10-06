from __future__ import annotations
import os, json, math, random, time, warnings
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
                             matthews_corrcoef, confusion_matrix, roc_auc_score, log_loss,
                             classification_report)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings('ignore')
SEED=42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(max(1,min(8,os.cpu_count() or 4)))
DEVICE=torch.device('cpu')
ROOT=Path(os.environ.get('ESWA_QA724_ROOT', Path(__file__).resolve().parents[1]))
DATA=ROOT/'Dataset/Dataset'
TRAIN_ROOT=DATA/'Training dataset/Otitis Media'
TEST_ROOT=DATA/'Testing dataset'
OUT=ROOT/'experiment_outputs'; OUT.mkdir(exist_ok=True)
CLASSES=['AOM','ASOM','CSOM','Normal']
CLASS_TO_IDX={c:i for i,c in enumerate(CLASSES)}
FOLDER_TO_CLASS={'AOM':'AOM','ASOM':'ASOM','CSOM':'CSOM','Normal Tympanic Membrane':'Normal'}

# Exact QA removals recovered from duplicate audit: 3 cross-label conflict pairs + 6 redundant same-label copies.
REMOVE_REL={
 'AOM/aom (11).png','Normal Tympanic Membrane/normal (53).png',
 'AOM/aom (8).png','Normal Tympanic Membrane/normal (51).png',
 'AOM/aom (9).png','Normal Tympanic Membrane/normal (52).png',
 'ASOM/asom (23).JPG','ASOM/asom (28).JPG','ASOM/asom (30).JPG',
 'ASOM/asom (41).JPG','ASOM/asom (75).JPG','ASOM/asom (7).JPG'
}

def load_crop(path:Path, size:int=96)->Image.Image:
    arr=np.fromfile(str(path),dtype=np.uint8)
    bgr=cv2.imdecode(arr,cv2.IMREAD_COLOR)
    if bgr is None: raise ValueError(f'unreadable: {path}')
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
    # visible field mask; retain central clinical field and remove black borders.
    mask=(gray>12).astype(np.uint8)*255
    n, labels, stats, cents=cv2.connectedComponentsWithStats(mask,8)
    if n>1:
        h,w=gray.shape; cy,cx=h/2,w/2
        best=None; bestscore=-1
        for i in range(1,n):
            x,y,ww,hh,area=stats[i]
            cxi,cyi=cents[i]
            center_pen=((cxi-cx)**2+(cyi-cy)**2)/(h*h+w*w)
            score=area*(1-0.6*center_pen)
            if score>bestscore: bestscore=score; best=(x,y,ww,hh)
        x,y,ww,hh=best
        pad=int(0.03*max(ww,hh)); x=max(0,x-pad); y=max(0,y-pad)
        x2=min(w,x+ww+2*pad); y2=min(h,y+hh+2*pad)
        rgb=rgb[y:y2,x:x2]
    im=Image.fromarray(rgb)
    im=ImageOps.fit(im,(size,size),method=Image.Resampling.LANCZOS,centering=(0.5,0.5))
    return im

expert=[]
for folder,c in FOLDER_TO_CLASS.items():
    for p in sorted((TRAIN_ROOT/folder).glob('*')):
        if p.is_file():
            rel=f'{folder}/{p.name}'
            if rel not in REMOVE_REL:
                expert.append({'path':p,'label':c,'source':'expert','rel':rel})
print('Expert cleaned',len(expert),Counter(x['label'] for x in expert))
assert len(expert)==687

# stratified 70/15/15 on cleaned expert cohort
idx=np.arange(len(expert)); y=np.array([CLASS_TO_IDX[x['label']] for x in expert])
train_idx,temp_idx=train_test_split(idx,test_size=0.30,random_state=SEED,stratify=y)
temp_y=y[temp_idx]
val_idx,test_idx=train_test_split(temp_idx,test_size=0.50,random_state=SEED,stratify=temp_y)
expert_train=[expert[i] for i in train_idx]; expert_val=[expert[i] for i in val_idx]; expert_test=[expert[i] for i in test_idx]

# accepted QA pseudo-labelled samples
pdf=pd.read_csv(ROOT/'test_ai_pseudo_labels_all_204.csv')
acc=pdf[pdf['included_in_QA_clean_724']==True]
pseudo=[]
for _,r in acc.iterrows():
    p=TEST_ROOT/r['original_filename']
    if not p.exists(): raise FileNotFoundError(p)
    pseudo.append({'path':p,'label':r['ai_predicted_label'],'source':'pseudo','confidence':float(r['ensemble_confidence'])})
assert len(pseudo)==37
print('Pseudo',Counter(x['label'] for x in pseudo))
print('splits',len(expert_train),len(expert_val),len(expert_test),Counter(x['label'] for x in expert_test))

# cache cropped arrays to accelerate repeated training
CACHE={}
def get_arr(path:Path,size:int):
    key=(str(path),size)
    if key not in CACHE:
        CACHE[key]=np.asarray(load_crop(path,size),dtype=np.uint8)
    return CACHE[key]

# VAE dataset and model (64x64)
class VAEDataset(Dataset):
    def __init__(self,records): self.records=records
    def __len__(self): return len(self.records)
    def __getitem__(self,i):
        r=self.records[i]; im=Image.fromarray(get_arr(r['path'],64))
        x=transforms.ToTensor()(im)
        return x,CLASS_TO_IDX[r['label']]

class CVAE(nn.Module):
    def __init__(self,latent=48,nc=4):
        super().__init__(); self.latent=latent; self.nc=nc
        self.enc=nn.Sequential(nn.Conv2d(3+nc,24,4,2,1),nn.BatchNorm2d(24),nn.LeakyReLU(.2),
                               nn.Conv2d(24,48,4,2,1),nn.BatchNorm2d(48),nn.LeakyReLU(.2),
                               nn.Conv2d(48,96,4,2,1),nn.BatchNorm2d(96),nn.LeakyReLU(.2),
                               nn.Conv2d(96,128,4,2,1),nn.BatchNorm2d(128),nn.LeakyReLU(.2))
        self.fc_mu=nn.Linear(128*4*4,latent); self.fc_lv=nn.Linear(128*4*4,latent)
        self.fc_dec=nn.Linear(latent+nc,128*4*4)
        self.dec=nn.Sequential(nn.ConvTranspose2d(128,96,4,2,1),nn.BatchNorm2d(96),nn.ReLU(),
                               nn.ConvTranspose2d(96,48,4,2,1),nn.BatchNorm2d(48),nn.ReLU(),
                               nn.ConvTranspose2d(48,24,4,2,1),nn.BatchNorm2d(24),nn.ReLU(),
                               nn.ConvTranspose2d(24,3,4,2,1),nn.Sigmoid())
    def encode(self,x,y):
        oh=F.one_hot(y,self.nc).float()[:,:,None,None].expand(-1,-1,64,64)
        h=self.enc(torch.cat([x,oh],1)).flatten(1)
        return self.fc_mu(h),self.fc_lv(h)
    def reparam(self,mu,lv): return mu+torch.randn_like(mu)*torch.exp(.5*lv)
    def decode(self,z,y):
        oh=F.one_hot(y,self.nc).float(); h=self.fc_dec(torch.cat([z,oh],1)).view(-1,128,4,4)
        return self.dec(h)
    def forward(self,x,y):
        mu,lv=self.encode(x,y); z=self.reparam(mu,lv); return self.decode(z,y),mu,lv

vae=CVAE().to(DEVICE)
opt=torch.optim.AdamW(vae.parameters(),lr=1.5e-3,weight_decay=1e-5)
loader=DataLoader(VAEDataset(expert_train),batch_size=32,shuffle=True,num_workers=0)
vae_hist=[]
start=time.time()
for ep in range(1,19):
    vae.train(); tot=0; recs=0; kls=0
    for x,yb in loader:
        x=x.to(DEVICE); yb=yb.to(DEVICE); out,mu,lv=vae(x,yb)
        rec=F.mse_loss(out,x,reduction='mean')
        kl=-0.5*torch.mean(1+lv-mu.pow(2)-lv.exp())
        beta=min(0.003,0.0005+ep*0.00015)
        loss=rec+beta*kl
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(vae.parameters(),5); opt.step()
        tot+=loss.item()*len(x); recs+=rec.item()*len(x); kls+=kl.item()*len(x)
    row={'epoch':ep,'loss':tot/len(loader.dataset),'reconstruction':recs/len(loader.dataset),'kl':kls/len(loader.dataset)}
    vae_hist.append(row)
    if ep in [1,6,12,18]: print('VAE',row)
torch.save(vae.state_dict(),OUT/'cvae_state.pt')
pd.DataFrame(vae_hist).to_csv(OUT/'cvae_training_history.csv',index=False)
print('VAE seconds',time.time()-start)

# encode class latent distributions and generate constrained samples
vae.eval(); latents={c:[] for c in CLASSES}
with torch.no_grad():
    for x,yb in DataLoader(VAEDataset(expert_train),batch_size=32,shuffle=False):
        mu,lv=vae.encode(x.to(DEVICE),yb.to(DEVICE))
        for z,yi in zip(mu.cpu(),yb): latents[CLASSES[int(yi)]].append(z.numpy())

train_counts=Counter(r['label'] for r in expert_train+pseudo)
target=120
synthetic=[]
with torch.no_grad():
    for c in ['AOM','ASOM','CSOM']:
        n=max(0,target-train_counts[c]); Z=np.stack(latents[c]); mean=Z.mean(0); std=Z.std(0)+1e-3
        # shrink towards observed class manifold to reduce implausible samples
        z=np.random.normal(mean,0.65*std,size=(n,Z.shape[1])).astype('float32')
        yb=torch.full((n,),CLASS_TO_IDX[c],dtype=torch.long)
        imgs=vae.decode(torch.from_numpy(z),yb).cpu().numpy()
        for j,img in enumerate(imgs):
            arr=(np.transpose(img,(1,2,0))*255).clip(0,255).astype(np.uint8)
            synthetic.append({'array':arr,'label':c,'source':'cVAE'})
print('Synthetic',Counter(x['label'] for x in synthetic),len(synthetic))
# save montage
import matplotlib.pyplot as plt
fig,axs=plt.subplots(3,6,figsize=(10,5.3))
for ri,c in enumerate(['AOM','ASOM','CSOM']):
    samples=[s for s in synthetic if s['label']==c][:6]
    for ci,s in enumerate(samples):
        axs[ri,ci].imshow(s['array']); axs[ri,ci].axis('off')
        if ci==0: axs[ri,ci].set_title(c,fontsize=9)
fig.suptitle('Conditional VAE synthetic minority-class samples (research augmentation only)',fontsize=11)
plt.tight_layout(); fig.savefig(OUT/'synthetic_montage.png',dpi=220,bbox_inches='tight'); plt.close(fig)

# classifier dataset
train_tf=transforms.Compose([
    transforms.RandomHorizontalFlip(p=.5),
    transforms.RandomRotation(12),
    transforms.RandomAffine(degrees=0,translate=(.06,.06),scale=(.92,1.08)),
    transforms.ColorJitter(brightness=.14,contrast=.14,saturation=.10,hue=.02),
    transforms.ToTensor(),
    transforms.Normalize([.5,.5,.5],[.25,.25,.25])])
eval_tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.5,.5,.5],[.25,.25,.25])])
class ImgDataset(Dataset):
    def __init__(self,records,train=False): self.records=records; self.tf=train_tf if train else eval_tf
    def __len__(self): return len(self.records)
    def __getitem__(self,i):
        r=self.records[i]
        if 'array' in r: im=Image.fromarray(r['array']).resize((96,96),Image.Resampling.LANCZOS)
        else: im=Image.fromarray(get_arr(r['path'],96))
        return self.tf(im),CLASS_TO_IDX[r['label']]

class ResBlock(nn.Module):
    def __init__(self,cin,cout,stride=1):
        super().__init__(); self.c1=nn.Conv2d(cin,cout,3,stride,1,bias=False); self.b1=nn.BatchNorm2d(cout)
        self.c2=nn.Conv2d(cout,cout,3,1,1,bias=False); self.b2=nn.BatchNorm2d(cout)
        self.skip=nn.Identity() if cin==cout and stride==1 else nn.Sequential(nn.Conv2d(cin,cout,1,stride,bias=False),nn.BatchNorm2d(cout))
    def forward(self,x): return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x)))))+self.skip(x))
class TinyResNet(nn.Module):
    def __init__(self,nc=4):
        super().__init__(); self.stem=nn.Sequential(nn.Conv2d(3,32,5,2,2,bias=False),nn.BatchNorm2d(32),nn.ReLU())
        self.body=nn.Sequential(ResBlock(32,32),ResBlock(32,64,2),ResBlock(64,64),ResBlock(64,128,2),ResBlock(128,128),ResBlock(128,192,2))
        self.pool=nn.AdaptiveAvgPool2d(1); self.drop=nn.Dropout(.30); self.fc=nn.Linear(192,nc)
    def forward(self,x): return self.fc(self.drop(self.pool(self.body(self.stem(x))).flatten(1)))

def evaluate(model,records,batch=64):
    model.eval(); probs=[]; ys=[]
    with torch.no_grad():
        for x,yb in DataLoader(ImgDataset(records,False),batch_size=batch,shuffle=False):
            p=F.softmax(model(x.to(DEVICE)),1).cpu().numpy(); probs.append(p); ys.append(yb.numpy())
    return np.concatenate(ys),np.vstack(probs)

def metrics(ys,probs):
    pred=probs.argmax(1); pr,rc,f1,_=precision_recall_fscore_support(ys,pred,average='macro',zero_division=0)
    try: auc=roc_auc_score(label_binarize(ys,classes=np.arange(4)),probs,average='macro',multi_class='ovr')
    except: auc=float('nan')
    return {'accuracy':accuracy_score(ys,pred),'balanced_accuracy':balanced_accuracy_score(ys,pred),
            'macro_precision':pr,'macro_recall':rc,'macro_f1':f1,'macro_auc':auc,
            'MCC':matthews_corrcoef(ys,pred),'log_loss':log_loss(ys,probs,labels=np.arange(4))}

def train_variant(name,records,seed=42,epochs=20):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model=TinyResNet().to(DEVICE)
    cnt=Counter(r['label'] for r in records)
    weights=torch.tensor([len(records)/(4*cnt[c]) for c in CLASSES],dtype=torch.float32)
    criterion=nn.CrossEntropyLoss(weight=weights,label_smoothing=.04)
    opt=torch.optim.AdamW(model.parameters(),lr=1.2e-3,weight_decay=2e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=8e-5)
    dl=DataLoader(ImgDataset(records,True),batch_size=32,shuffle=True,num_workers=0)
    best=None; best_score=-1; patience=6; wait=0; hist=[]
    for ep in range(1,epochs+1):
        model.train(); losses=[]
        for x,yb in dl:
            logits=model(x.to(DEVICE)); loss=criterion(logits,yb.to(DEVICE)); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5); opt.step(); losses.append(loss.item())
        sched.step(); vy,vp=evaluate(model,expert_val); vm=metrics(vy,vp)
        hist.append({'epoch':ep,'train_loss':np.mean(losses),**{'val_'+k:v for k,v in vm.items()}})
        score=vm['macro_f1']
        if score>best_score+1e-4:
            best_score=score; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; wait=0
        else: wait+=1
        if ep in [1,5,10,15,20] or wait>=patience: print(name,ep,round(np.mean(losses),4),{k:round(vm[k],3) for k in ['accuracy','balanced_accuracy','macro_f1','macro_auc']})
        if wait>=patience and ep>=10: break
    model.load_state_dict(best)
    ty,tp=evaluate(model,expert_test); tm=metrics(ty,tp)
    torch.save(model.state_dict(),OUT/f'{name}_model.pt')
    pd.DataFrame(hist).to_csv(OUT/f'{name}_history.csv',index=False)
    np.savez(OUT/f'{name}_test_predictions.npz',y=ty,probs=tp)
    return model,tm,ty,tp,hist

variants={
 'Expert-only CNN': expert_train,
 'QA-724 CNN': expert_train+pseudo,
 'GenAI-augmented CNN': expert_train+pseudo+synthetic,
}
results=[]; pred_store={}; models={}
for name,recs in variants.items():
    print('\nTRAIN',name,'n=',len(recs),Counter(r['label'] for r in recs))
    model,m,ytrue,prob,hist=train_variant(name.replace(' ','_').replace('-','_'),recs,SEED,22)
    results.append({'model':name,'training_samples':len(recs),**m}); pred_store[name]=(ytrue,prob); models[name]=model

res=pd.DataFrame(results); res.to_csv(OUT/'deep_learning_model_comparison.csv',index=False)
print(res.to_string(index=False))

# Temperature scaling final GenAI model using validation logits
final=models['GenAI-augmented CNN']
final.eval()
def get_logits(records):
    ls=[]; ys=[]
    with torch.no_grad():
        for x,yb in DataLoader(ImgDataset(records,False),batch_size=64): ls.append(final(x).cpu()); ys.append(yb)
    return torch.cat(ls),torch.cat(ys)
vl,vy=get_logits(expert_val); tl,ty=get_logits(expert_test)
logT=torch.tensor(0.0,requires_grad=True)
optT=torch.optim.LBFGS([logT],lr=.1,max_iter=80)
def closure():
    optT.zero_grad(); loss=F.cross_entropy(vl/torch.exp(logT),vy); loss.backward(); return loss
optT.step(closure); T=float(torch.exp(logT).detach())
cal_probs=F.softmax(tl/T,1).detach().numpy(); cal_m=metrics(ty.numpy(),cal_probs)
np.savez(OUT/'final_calibrated_predictions.npz',y=ty.numpy(),probs=cal_probs,temperature=T)
with open(OUT/'temperature_calibration.json','w') as f: json.dump({'temperature':T,'metrics':cal_m},f,indent=2)
print('Temperature',T,'cal metrics',cal_m)

# referral/selective prediction: entropy + max confidence; report coverage-risk curve
entropy=-(cal_probs*np.log(np.clip(cal_probs,1e-9,1))).sum(1)/math.log(4)
conf=cal_probs.max(1); pred=cal_probs.argmax(1); correct=(pred==ty.numpy()).astype(float)
rows=[]
for threshold in np.linspace(.25,.90,14):
    auto=(conf>=threshold)&(entropy<=0.65)
    coverage=auto.mean(); auto_acc=correct[auto].mean() if auto.any() else np.nan
    referral=1-coverage
    rows.append({'confidence_threshold':threshold,'coverage':coverage,'referral_rate':referral,'retained_accuracy':auto_acc,'retained_n':int(auto.sum())})
sel=pd.DataFrame(rows); sel.to_csv(OUT/'selective_referral_analysis.csv',index=False)

# confusion matrix, class report, AUCs final
cm=confusion_matrix(ty.numpy(),pred,labels=np.arange(4)); pd.DataFrame(cm,index=CLASSES,columns=CLASSES).to_csv(OUT/'final_confusion_matrix.csv')
report=classification_report(ty.numpy(),pred,target_names=CLASSES,output_dict=True,zero_division=0); pd.DataFrame(report).T.to_csv(OUT/'final_classification_report.csv')
aucs={}
for i,c in enumerate(CLASSES): aucs[c]=roc_auc_score((ty.numpy()==i).astype(int),cal_probs[:,i])
with open(OUT/'final_class_auc.json','w') as f: json.dump(aucs,f,indent=2)

# bootstrap distributions for CI/box/violin
rng=np.random.default_rng(42); boot=[]; n=len(ty)
for b in range(1000):
    ix=rng.integers(0,n,n); yy=ty.numpy()[ix]; pp=cal_probs[ix];
    if len(np.unique(yy))<4: continue
    mm=metrics(yy,pp); boot.append({'replicate':b,**mm})
pd.DataFrame(boot).to_csv(OUT/'bootstrap_metrics_1000.csv',index=False)

# save data manifest/split summary
manifest=[]
for split,recs in [('train_expert',expert_train),('validation_expert',expert_val),('test_expert',expert_test),('train_pseudo',pseudo)]:
    for r in recs: manifest.append({'split':split,'path':str(r['path']),'label':r['label'],'source':r['source']})
pd.DataFrame(manifest).to_csv(OUT/'experiment_manifest.csv',index=False)
summary={'qa_clean_total':724,'expert_clean':687,'pseudo_added':37,'expert_train':len(expert_train),'validation':len(expert_val),'test':len(expert_test),
         'class_counts_qa724':dict(Counter(r['label'] for r in expert+pseudo)),'removed_files':sorted(REMOVE_REL),'synthetic_count':len(synthetic)}
with open(OUT/'experiment_summary.json','w') as f: json.dump(summary,f,indent=2)
print('DONE',OUT)
