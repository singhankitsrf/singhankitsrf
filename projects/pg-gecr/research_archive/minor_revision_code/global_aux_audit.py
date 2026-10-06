from pathlib import Path
from collections import defaultdict,Counter
import hashlib,json,random,csv
import numpy as np
from PIL import Image
from scipy.fftpack import dct
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score,f1_score,balanced_accuracy_score,matthews_corrcoef,roc_auc_score,confusion_matrix,classification_report,average_precision_score
from sklearn.preprocessing import label_binarize

ROOT=Path('/mnt/data/minor_work/auxdata/Otoscopic_Data')
OUT=Path('/mnt/data/minor_work/minor_results');OUT.mkdir(exist_ok=True)
CLASSES=['Acute Otitis Media','Cerumen Impaction','Chronic Otitis Media','Myringosclerosis','Normal']; C2I={c:i for i,c in enumerate(CLASSES)}
SEED=20260927

def phash(path):
    im=Image.open(path).convert('L').resize((32,32),Image.Resampling.LANCZOS)
    arr=np.asarray(im,dtype=float); dc=dct(dct(arr,axis=0),axis=1)[:8,:8];med=np.median(dc);v=0
    for b in (dc>med).flatten():v=(v<<1)|int(b)
    return v
# inventory + global exact hash audit
raw=[]
for c in CLASSES:
    for p in sorted((ROOT/c).glob('*.jpg')):raw.append((p,c))
sha=defaultdict(list)
for p,c in raw:sha[hashlib.sha256(p.read_bytes()).hexdigest()].append((p,c))
exact_cross=[v for v in sha.values() if len({c for _,c in v})>1]
# Exclude cross-class exact conflicts; retain one deterministic representative per same-class exact group
uniq=[];sameclass_redundant=0
for h,v in sorted(sha.items()):
    labs={c for _,c in v}
    if len(labs)>1:continue
    v=sorted(v,key=lambda x:str(x[0]));uniq.append(v[0]);sameclass_redundant+=len(v)-1
print('raw',len(raw),'unique retained',len(uniq),'sameclass redundant',sameclass_redundant,'cross exact groups',len(exact_cross),flush=True)
# pHashes globally
hs=[]
for i,(p,c) in enumerate(uniq):
    hs.append(phash(p))
    if (i+1)%500==0:print('phash',i+1,flush=True)
n=len(uniq);par=list(range(n));sz=[1]*n

def find(a):
    while par[a]!=a:par[a]=par[par[a]];a=par[a]
    return a
def union(a,b):
    a,b=find(a),find(b)
    if a==b:return
    if sz[a]<sz[b]:a,b=b,a
    par[b]=a;sz[a]+=sz[b]
# global pair graph
cross_edges=[];edge_count=0
for i in range(n):
    hi=hs[i]; ci=uniq[i][1]
    for j in range(i+1,n):
        if (hi^hs[j]).bit_count()<=4:
            edge_count+=1;union(i,j)
            if ci!=uniq[j][1]:cross_edges.append((i,j,(hi^hs[j]).bit_count()))
    if (i+1)%500==0:print('pairs',i+1,'edges',edge_count,'cross',len(cross_edges),flush=True)
comp=defaultdict(list)
for i in range(n):comp[find(i)].append(i)
conflict=[];clean=[]
for ids in comp.values():
    labs={uniq[i][1] for i in ids}
    if len(labs)>1:conflict.append(ids)
    else:clean.append(ids)
print('components',len(comp),'conflict comps',len(conflict),'cross edges',len(cross_edges),'conflict images',sum(map(len,conflict)),flush=True)
# conflict details
conf_rows=[]
for gid,ids in enumerate(sorted(conflict,key=lambda x:(-len(x),min(x)))):
    labs=Counter(uniq[i][1] for i in ids)
    conf_rows.append({'conflict_component':gid,'size':len(ids),'classes':' | '.join(f'{k}:{v}' for k,v in sorted(labs.items())),'files':' | '.join(str(uniq[i][0].relative_to(ROOT)) for i in ids[:20])})
pd_conf=conf_rows
with open(OUT/'aux_global_crossclass_conflicts.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['conflict_component','size','classes','files']);w.writeheader();w.writerows(conf_rows)
# group homogeneous clean comps by class
byclass=defaultdict(list)
for ids in clean:
    c=uniq[ids[0]][1]; byclass[c].append(ids)
# deterministic split with same target val/test as prior where possible
def choose_groups_exact(groups,target,rng):
    idx=list(range(len(groups)));rng.shuffle(idx);dp={0:[]}
    for gi in idx:
        s=len(groups[gi])
        for cur,sel in list(dp.items())[::-1]:
            ns=cur+s
            if ns<=target and ns not in dp:dp[ns]=sel+[gi]
        if target in dp:break
    best=target if target in dp else max(dp);return set(dp[best]),best
rng=random.Random(SEED);split={'train':[],'val':[],'test':[]};audit={}
for c in CLASSES:
    groups=byclass[c]; total=sum(len(g) for g in groups); target=78 if c=='Acute Otitis Media' else 90
    target=min(target,max(1,int(round(total*.15)))) if total < 3*target else target
    gtest,nt=choose_groups_exact(groups,target,rng);remain=[g for i,g in enumerate(groups) if i not in gtest]
    gval,nv=choose_groups_exact(remain,target,rng)
    test=[i for gi,g in enumerate(groups) if gi in gtest for i in g]
    val=[i for gi,g in enumerate(remain) if gi in gval for i in g]
    train=[i for gi,g in enumerate(remain) if gi not in gval for i in g]
    for i in train:split['train'].append((uniq[i][0],C2I[c]))
    for i in val:split['val'].append((uniq[i][0],C2I[c]))
    for i in test:split['test'].append((uniq[i][0],C2I[c]))
    audit[c]={'post_conflict_images':total,'global_components':len(groups),'largest_group':max(map(len,groups)) if groups else 0,'train':len(train),'val':len(val),'test':len(test)}
print('split audit',audit,flush=True)
# verify no clean component crosses splits
path_split={str(p):sp for sp,items in split.items() for p,y in items}
viol=0
for ids in clean:
    s={path_split.get(str(uniq[i][0])) for i in ids};s.discard(None)
    if len(s)>1:viol+=1
assert viol==0
# manifest
with open(OUT/'aux_global_split_manifest.csv','w',newline='') as f:
    w=csv.writer(f);w.writerow(['split','class','label','file'])
    for sp in ['train','val','test']:
        for p,y in split[sp]:w.writerow([sp,CLASSES[y],y,str(p.relative_to(ROOT))])
# prepare fixed arrays
H=W=64;Yg,Xg=np.mgrid[0:H,0:W];cx=(W-1)/2;cy=(H-1)/2;dist=np.sqrt(((Xg-cx)/(0.47*W))**2+((Yg-cy)/(0.47*H))**2);mask=np.clip((1.08-dist)/0.14,0,1).astype(np.float32)[...,None]
def load(items):
 xs=[];ys=[]
 for p,y in items:
    a=np.asarray(Image.open(p).convert('RGB').resize((W,H),Image.Resampling.BILINEAR),dtype=np.float32)/255.;a=a*mask;xs.append(a.transpose(2,0,1));ys.append(y)
 return np.stack(xs).astype(np.float32),np.array(ys,np.int64)
Xtr,ytr=load(split['train']);Xv,yv=load(split['val']);Xt,yt=load(split['test'])
def features(X):return np.concatenate([X[:,:,::4,::4].reshape(len(X),-1),X.mean((2,3)),X.std((2,3))],1).astype(np.float32)
Ftr,Fv,Ft=features(Xtr),features(Xv),features(Xt)
def model(seed=20260921):return ExtraTreesClassifier(n_estimators=300,max_features=.5,class_weight='balanced',n_jobs=-1,random_state=seed)
m=model();m.fit(Ftr,ytr);pv=m.predict(Fv);pp=m.predict(Ft);pro=m.predict_proba(Ft)
metrics={'accuracy':float(accuracy_score(yt,pp)),'macro_f1':float(f1_score(yt,pp,average='macro',zero_division=0)),'balanced_accuracy':float(balanced_accuracy_score(yt,pp)),'mcc':float(matthews_corrcoef(yt,pp)),'macro_auc':float(roc_auc_score(yt,pro,multi_class='ovr',average='macro')),'val_macro_f1':float(f1_score(yv,pv,average='macro',zero_division=0)),'n_train':len(ytr),'n_val':len(yv),'n_test':len(yt)}
np.savetxt(OUT/'aux_global_confusion.csv',confusion_matrix(yt,pp),delimiter=',',fmt='%d')
pd=np
# classwise report
rep=classification_report(yt,pp,target_names=CLASSES,output_dict=True,zero_division=0);Yb=label_binarize(yt,classes=np.arange(5));rows=[]
for ci,c in enumerate(CLASSES):rows.append({'class':c,'precision':rep[c]['precision'],'recall':rep[c]['recall'],'f1':rep[c]['f1-score'],'support':int(rep[c]['support']),'auc':float(roc_auc_score(Yb[:,ci],pro[:,ci])),'ap':float(average_precision_score(Yb[:,ci],pro[:,ci]))})
with open(OUT/'aux_global_classwise.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
# repeated balanced training-subset learning curve; fixed test/val, same ExtraTrees hyperparams; 10 seeds per size
curve=[]
for t in [25,50,100,200,350]:
  for repid in range(10):
    rg=np.random.default_rng(SEED + 1000*t + repid);ids=[]
    for c in range(5):
      ix=np.where(ytr==c)[0].copy();rg.shuffle(ix);ids.extend(ix[:min(t,len(ix))].tolist())
    ids=np.array(ids);rg.shuffle(ids);mm=model(seed=20260921) # model RNG held fixed; only subset selection changes
    mm.fit(Ftr[ids],ytr[ids]);vpred=mm.predict(Fv);tpred=mm.predict(Ft);tpro=mm.predict_proba(Ft)
    curve.append({'per_class_training':t,'repeat':repid,'subset_seed':SEED+1000*t+repid,'n_train':len(ids),'val_macro_f1':float(f1_score(yv,vpred,average='macro',zero_division=0)),'test_macro_f1':float(f1_score(yt,tpred,average='macro',zero_division=0)),'test_accuracy':float(accuracy_score(yt,tpred)),'test_balanced_accuracy':float(balanced_accuracy_score(yt,tpred)),'test_macro_auc':float(roc_auc_score(yt,tpro,multi_class='ovr',average='macro'))})
  print('curve',t,flush=True)
with open(OUT/'aux_global_learning_curve_repeated.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=curve[0].keys());w.writeheader();w.writerows(curve)
# aggregate mean/sd/range
agg=[]
for t in [25,50,100,200,350]:
  arr=[r for r in curve if r['per_class_training']==t]
  vals=np.array([r['test_macro_f1'] for r in arr]);vacc=np.array([r['test_accuracy'] for r in arr]);vauc=np.array([r['test_macro_auc'] for r in arr])
  agg.append({'per_class_training':t,'repeats':len(arr),'macro_f1_mean':float(vals.mean()),'macro_f1_sd':float(vals.std(ddof=1)),'macro_f1_min':float(vals.min()),'macro_f1_max':float(vals.max()),'accuracy_mean':float(vacc.mean()),'accuracy_sd':float(vacc.std(ddof=1)),'macro_auc_mean':float(vauc.mean()),'macro_auc_sd':float(vauc.std(ddof=1))})
with open(OUT/'aux_global_learning_curve_summary.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=agg[0].keys());w.writeheader();w.writerows(agg)
summary={'raw_images':len(raw),'same_class_exact_redundant_removed':sameclass_redundant,'cross_class_exact_conflict_groups':len(exact_cross),'exact_unique_retained_before_phash':len(uniq),'global_phash_edges':edge_count,'cross_class_phash_edges':len(cross_edges),'global_phash_components':len(comp),'cross_class_conflict_components':len(conflict),'cross_class_conflict_images_excluded':sum(map(len,conflict)),'post_conflict_images':sum(len(g) for g in clean),'split_audit':audit,'split_leakage_components':viol,'fixed_pipeline_metrics':metrics,'learning_curve_summary':agg}
json.dump(summary,open(OUT/'aux_global_audit_summary.json','w'),indent=2)
print(json.dumps(summary,indent=2))
