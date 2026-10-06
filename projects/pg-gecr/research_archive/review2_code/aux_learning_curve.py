import os, hashlib, json, random, time
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
from scipy.fftpack import dct
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, matthews_corrcoef

ROOT=Path('/mnt/data/rev2work/auxdata/Otoscopic_Data')
OUT=Path('/mnt/data/rev2work/aux_learning'); OUT.mkdir(exist_ok=True)
SEED=20260921
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(5)
classes=['Acute Otitis Media','Cerumen Impaction','Chronic Otitis Media','Myringosclerosis','Normal']
label={c:i for i,c in enumerate(classes)}

def phash(path):
    im=Image.open(path).convert('L').resize((32,32),Image.Resampling.LANCZOS)
    arr=np.asarray(im,dtype=float)
    dc=dct(dct(arr,axis=0),axis=1)[:8,:8]
    med=np.median(dc)
    v=0
    for b in (dc>med).flatten(): v=(v<<1)|int(b)
    return v

def groups_for(files):
    hs=[phash(p) for p in files]; n=len(files); par=list(range(n)); sz=[1]*n
    def f(a):
        while par[a]!=a: par[a]=par[par[a]]; a=par[a]
        return a
    def u(a,b):
        a,b=f(a),f(b)
        if a==b:return
        if sz[a]<sz[b]:a,b=b,a
        par[b]=a; sz[a]+=sz[b]
    for i in range(n):
        hi=hs[i]
        for j in range(i+1,n):
            if (hi^hs[j]).bit_count()<=4:u(i,j)
    d=defaultdict(list)
    for i,p in enumerate(files):d[f(i)].append(p)
    return list(d.values())

def choose_groups_exact(groups,target,rng):
    idx=list(range(len(groups))); rng.shuffle(idx)
    # DP sum -> chosen tuple; randomized group order creates deterministic random solution
    dp={0:[]}
    for gi in idx:
        s=len(groups[gi])
        for cur,sel in list(dp.items())[::-1]:
            ns=cur+s
            if ns<=target and ns not in dp: dp[ns]=sel+[gi]
        if target in dp: break
    if target not in dp:
        # closest below target
        best=max(dp)
        return set(dp[best]),best
    return set(dp[target]),target

# exact dedupe then groups then split
split={'train':[],'val':[],'test':[]}; audit={}
rng=random.Random(SEED)
for c in classes:
    raw=sorted((ROOT/c).glob('*.jpg'))
    sh={}
    for p in raw:
        h=hashlib.sha256(p.read_bytes()).hexdigest()
        if h not in sh: sh[h]=p
    uniq=list(sh.values()); uniq=sorted(uniq,key=lambda p:p.name)
    groups=groups_for(uniq)
    target_test=78 if c=='Acute Otitis Media' else 90
    target_val=78 if c=='Acute Otitis Media' else 90
    gtest,nt=choose_groups_exact(groups,target_test,rng)
    remain=[g for i,g in enumerate(groups) if i not in gtest]
    gval,nv=choose_groups_exact(remain,target_val,rng)
    test=[p for i,g in enumerate(groups) if i in gtest for p in g]
    val=[p for i,g in enumerate(remain) if i in gval for p in g]
    train=[p for i,g in enumerate(remain) if i not in gval for p in g]
    for p in train: split['train'].append((p,label[c]))
    for p in val: split['val'].append((p,label[c]))
    for p in test: split['test'].append((p,label[c]))
    audit[c]={'archive':len(raw),'unique':len(uniq),'groups':len(groups),'largest_group':max(map(len,groups)),'train':len(train),'val':len(val),'test':len(test)}
print('audit',audit)
json.dump(audit,open(OUT/'split_audit.json','w'),indent=2)
# manifest
with open(OUT/'split_manifest.csv','w') as f:
    f.write('split,class,label,file\n')
    for sp in ['train','val','test']:
        for p,y in split[sp]:f.write(f'{sp},{classes[y]},{y},{p}\n')

# preload arrays
H=W=64
# soft radial mask directly at 64
Y,X=np.mgrid[0:H,0:W]; cx=(W-1)/2; cy=(H-1)/2
dist=np.sqrt(((X-cx)/(0.47*W))**2+((Y-cy)/(0.47*H))**2)
mask=np.clip((1.08-dist)/0.14,0,1).astype(np.float32)[...,None]
def load_items(items):
    xs=[]; ys=[]
    for p,y in items:
        im=Image.open(p).convert('RGB').resize((W,H),Image.Resampling.BILINEAR)
        a=np.asarray(im,dtype=np.float32)/255.0
        a=a*mask
        xs.append(a.transpose(2,0,1)); ys.append(y)
    return np.stack(xs).astype(np.float32),np.array(ys,np.int64)
cache=OUT/'arrays.npz'
if cache.exists():
    z=np.load(cache); Xtr,ytr,Xv,yv,Xt,yt=[z[k] for k in ['Xtr','ytr','Xv','yv','Xt','yt']]
else:
    Xtr,ytr=load_items(split['train']); Xv,yv=load_items(split['val']); Xt,yt=load_items(split['test'])
    np.savez_compressed(cache,Xtr=Xtr,ytr=ytr,Xv=Xv,yv=yv,Xt=Xt,yt=yt)
print('arrays',Xtr.shape,Xv.shape,Xt.shape)

class ArrDS(Dataset):
    def __init__(self,X,y,aug=False): self.X=X;self.y=y;self.aug=aug
    def __len__(self):return len(self.y)
    def __getitem__(self,i):
        x=torch.from_numpy(self.X[i].copy()); y=int(self.y[i])
        if self.aug:
            if torch.rand(())<.5:x=torch.flip(x,[2])
            gain=0.9+0.2*torch.rand(())
            x=torch.clamp(x*gain,0,1)
            if torch.rand(())<.25:x=torch.clamp(x+torch.randn_like(x)*0.01,0,1)
        return x,y
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        def b(a,b):return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(),nn.MaxPool2d(2))
        self.features=nn.Sequential(b(3,24),b(24,48),b(48,96),nn.AdaptiveAvgPool2d(1))
        self.drop=nn.Dropout(.25); self.fc=nn.Linear(96,5)
    def forward(self,x):return self.fc(self.drop(self.features(x).flatten(1)))

def metrics(y,p):
    return {'accuracy':float(accuracy_score(y,p)),'macro_f1':float(f1_score(y,p,average='macro',zero_division=0)),'balanced_accuracy':float(balanced_accuracy_score(y,p)),'mcc':float(matthews_corrcoef(y,p))}

def run(target):
    # balanced deterministic subset target per class from fixed train pool
    inds=[]; rg=np.random.default_rng(SEED+target)
    for c in range(5):
        ix=np.where(ytr==c)[0]; rg.shuffle(ix); inds.extend(ix[:min(target,len(ix))].tolist())
    inds=np.array(inds); rg.shuffle(inds)
    Xt0=Xtr[inds]; yt0=ytr[inds]
    torch.manual_seed(SEED+target); np.random.seed(SEED+target); random.seed(SEED+target)
    model=Net(); opt=torch.optim.AdamW(model.parameters(),lr=1.4e-3,weight_decay=2e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=20)
    lossfn=nn.CrossEntropyLoss(label_smoothing=.04)
    tr=DataLoader(ArrDS(Xt0,yt0,True),batch_size=128,shuffle=True,num_workers=0)
    va=DataLoader(ArrDS(Xv,yv,False),batch_size=256,shuffle=False)
    te=DataLoader(ArrDS(Xt,yt,False),batch_size=256,shuffle=False)
    best=-1; beststate=None; bestep=0; patience=6; bad=0
    hist=[]
    for ep in range(1,31):
        model.train(); tot=0
        for xb,yb in tr:
            opt.zero_grad(); z=model(xb); L=lossfn(z,yb); L.backward();opt.step();tot+=L.item()*len(yb)
        sched.step()
        model.eval(); vp=[]
        with torch.no_grad():
            for xb,yb in va:vp.extend(model(xb).argmax(1).numpy().tolist())
        vf=f1_score(yv,np.array(vp),average='macro',zero_division=0)
        hist.append([ep,tot/len(yt0),vf])
        if vf>best+1e-6:
            best=vf;bestep=ep;beststate={k:v.detach().clone() for k,v in model.state_dict().items()};bad=0
        else: bad+=1
        if bad>=patience and ep>=10:break
    model.load_state_dict(beststate);model.eval(); pp=[]
    with torch.no_grad():
        for xb,yb in te:pp.extend(model(xb).argmax(1).numpy().tolist())
    m=metrics(yt,np.array(pp));m.update({'per_class_target':target,'n_train':len(inds),'best_epoch':bestep,'val_macro_f1':float(best)})
    pd=None
    np.savetxt(OUT/f'pred_target_{target}.csv',np.column_stack([yt,np.array(pp)]),delimiter=',',header='y_true,y_pred',comments='',fmt='%d')
    np.savetxt(OUT/f'history_target_{target}.csv',np.array(hist),delimiter=',',header='epoch,train_loss,val_macro_f1',comments='')
    json.dump(m,open(OUT/f'metrics_target_{target}.json','w'),indent=2)
    print('TARGET',target,m,flush=True)
    return m

if __name__=='__main__':
 import sys
 targets=[int(x) for x in sys.argv[1:]] if len(sys.argv)>1 else [25,50,100,200,350]
 out=[]
 for t in targets: out.append(run(t))
 # merge existing
 allm=[]
 for p in sorted(OUT.glob('metrics_target_*.json')): allm.append(json.load(open(p)))
 allm.sort(key=lambda d:d['per_class_target'])
 json.dump(allm,open(OUT/'learning_curve.json','w'),indent=2)
