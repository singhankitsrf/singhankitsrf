from pathlib import Path
import cv2, numpy as np, hashlib
from collections import defaultdict
root=Path('/mnt/data/eswa_genai_project/Dataset/Dataset/Training dataset/Otitis Media')
files=sorted([p for p in root.rglob('*') if p.is_file()])

def phash(path):
    data=np.fromfile(str(path),dtype=np.uint8)
    img=cv2.imdecode(data,cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    img=cv2.resize(img,(32,32),interpolation=cv2.INTER_AREA).astype(np.float32)
    dct=cv2.dct(img)
    block=dct[:8,:8]
    med=np.median(block[1:,:])
    bits=(block>med).flatten()
    out=0
    for b in bits: out=(out<<1)|int(b)
    return out

def ham(a,b): return (a^b).bit_count()

hashes={p:phash(p) for p in files}
# exact file hashes
sha=defaultdict(list)
for p in files:
    sha[hashlib.sha256(p.read_bytes()).hexdigest()].append(p)
print('EXACT')
for v in sha.values():
    if len(v)>1: print([(str(p.relative_to(root)), p.stat().st_size) for p in v])
print('NEAR <=2')
pairs=[]
for i,p in enumerate(files):
  for q in files[i+1:]:
    d=ham(hashes[p],hashes[q])
    if d<=2:
      pairs.append((d,p.relative_to(root),q.relative_to(root)))
for x in pairs: print(x)
print('count',len(pairs))
