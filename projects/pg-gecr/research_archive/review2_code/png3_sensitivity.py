from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, roc_auc_score, matthews_corrcoef, log_loss, average_precision_score

SEED=42
TRAIN=Path('/mnt/data/eswa_final_work/Dataset/Training dataset/Otitis Media')
CLASSES=['AOM','ASOM','CSOM','Normal']; C2I={c:i for i,c in enumerate(CLASSES)}
F2C={'AOM':'AOM','ASOM':'ASOM','CSOM':'CSOM','Normal Tympanic Membrane':'Normal'}
REMOVE={'AOM/aom (11).png','Normal Tympanic Membrane/normal (53).png','AOM/aom (8).png','Normal Tympanic Membrane/normal (51).png','AOM/aom (9).png','Normal Tympanic Membrane/normal (52).png','ASOM/asom (23).JPG','ASOM/asom (28).JPG','ASOM/asom (30).JPG','ASOM/asom (41).JPG','ASOM/asom (75).JPG','ASOM/asom (7).JPG'}
expert=[]
for folder,c in F2C.items():
    for p in sorted((TRAIN/folder).glob('*')):
        rel=f'{folder}/{p.name}'
        if p.is_file() and rel not in REMOVE: expert.append({'path':p,'label':c,'rel':rel})
y=np.array([C2I[r['label']] for r in expert]); idx=np.arange(len(expert))
tr,tmp=train_test_split(idx,test_size=.30,stratify=y,random_state=SEED); va,te=train_test_split(tmp,test_size=.5,stratify=y[tmp],random_state=SEED)
z=np.load('/mnt/data/eswa_final_work/roi_results_fast/features.npz'); X=z['X']
# first 687 features order is trr,var,ter, not expert natural order; recreate mapping order
records=[expert[i] for i in tr]+[expert[i] for i in va]+[expert[i] for i in te]
a,b,c=len(tr),len(tr)+len(va),len(tr)+len(va)+len(te)
assert c==687
# indices within records corresponding split ranges
split={'train':np.arange(0,a),'val':np.arange(a,b),'test':np.arange(b,c)}
# Keep PNG and labels AOM,CSOM,Normal; map 0,2,3 -> 0,1,2
keep_labels={'AOM':0,'CSOM':1,'Normal':2}
def select(which):
    inds=[]; yy=[]; names=[]
    for k in split[which]:
        r=records[int(k)]
        if r['label'] in keep_labels and r['path'].suffix.lower()=='.png':
            inds.append(int(k)); yy.append(keep_labels[r['label']]); names.append(r['rel'])
    return np.array(inds),np.array(yy),names
itr,ytr,ntr=select('train'); iv,yv,nv=select('val'); it,yt,nt=select('test')
model=Pipeline([('s',StandardScaler()),('p',PCA(n_components=.97,whiten=True,random_state=SEED)),('m',LogisticRegression(C=.05,max_iter=4000,class_weight='balanced',random_state=SEED))])
model.fit(X[itr],ytr); pv=model.predict_proba(X[iv]); pt=model.predict_proba(X[it]); pred=pt.argmax(1)
# no post-review hyperparameter tuning; fixed C from expert primary branch
classes=['AOM','CSOM','Normal']
rep=classification_report(yt,pred,target_names=classes,output_dict=True,zero_division=0)
Ybin=label_binarize(yt,classes=np.arange(3))
aucs={classes[i]:roc_auc_score(Ybin[:,i],pt[:,i]) for i in range(3)}
aps={classes[i]:average_precision_score(Ybin[:,i],pt[:,i]) for i in range(3)}
macro_auc=roc_auc_score(Ybin,pt,average='macro',multi_class='ovr')
metrics={
 'train_n':len(ytr),'val_n':len(yv),'test_n':len(yt),
 'train_counts':{c:int((ytr==i).sum()) for i,c in enumerate(classes)},
 'val_counts':{c:int((yv==i).sum()) for i,c in enumerate(classes)},
 'test_counts':{c:int((yt==i).sum()) for i,c in enumerate(classes)},
 'accuracy':accuracy_score(yt,pred),'balanced_accuracy':balanced_accuracy_score(yt,pred),
 'macro_f1':precision_recall_fscore_support(yt,pred,average='macro',zero_division=0)[2],
 'macro_auc':macro_auc,'MCC':matthews_corrcoef(yt,pred),'log_loss':log_loss(yt,pt,labels=np.arange(3)),
 'class_auc':aucs,'class_ap':aps,
}
print(metrics)
print(pd.DataFrame(rep).T)
print(pd.DataFrame(confusion_matrix(yt,pred),index=classes,columns=classes))
out=Path('/mnt/data/rev2work/results'); out.mkdir(exist_ok=True)
pd.DataFrame(rep).T.to_csv(out/'png3_classification_report.csv')
pd.DataFrame(confusion_matrix(yt,pred),index=classes,columns=classes).to_csv(out/'png3_confusion_matrix.csv')
pd.DataFrame({'filename':nt,'true':[classes[i] for i in yt],'pred':[classes[i] for i in pred],**{f'p_{c}':pt[:,i] for i,c in enumerate(classes)}}).to_csv(out/'png3_test_predictions.csv',index=False)
import json
json.dump(metrics,open(out/'png3_metrics.json','w'),indent=2)
