"""ROI extractor adapted from the archived study; original functions below."""
import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage.feature import hog, local_binary_pattern

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
