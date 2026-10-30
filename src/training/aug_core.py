"""Augmentation primitives copied verbatim from setup_20261027/imx27_ref63_2359/prepare_sixclass.py
(same optical/crop recipe as all previous IMX runs). Do not edit; build_aug.py checks the source hash."""
import hashlib,json
from pathlib import Path
import cv2
import numpy as np
from pycocotools import mask as mu

CLASSES=['water','bolt','nut','oil','paint','reflection']
CATEGORIES=[dict(id=n+1,name=c) for n,c in enumerate(CLASSES)]
TARGETS={1,2,3,4,5}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):Path(p).write_text(json.dumps(value,ensure_ascii=False,indent=2))
def encode(m):
 r=mu.encode(np.asfortranarray(m.astype(np.uint8)));r['counts']=r['counts'].decode();return r
def decode(r):
 if isinstance(r['counts'],list):r=mu.frPyObjects(r,*r['size'])
 return mu.decode(r).astype(np.uint8)
def annotation(mask,cid,oid):
 y,x=np.nonzero(mask);assert len(x)
 return dict(category_id=cid,source_object_id=oid,segmentation=encode(mask),area=int(mask.sum()),
             bbox=[int(x.min()),int(y.min()),int(x.max()-x.min()+1),int(y.max()-y.min()+1)],iscrowd=0)
def raw_mask(obj,w,h):
 if 'mask' in obj:m=np.ascontiguousarray(decode(obj['mask']))
 else:m=np.zeros((h,w),np.uint8)
 for r in obj['regions']:cv2.fillPoly(m,[np.rint(r['points']).astype(np.int32)],0 if r['subtract'] else 1)
 return m
def transform_boxes(boxes,window,flip):
 if boxes is None:return None
 x,y,w,h=window;result=[]
 for a,b,c,d in boxes:
  a,b,c,d=max(0,a-x),max(0,b-y),min(w,c-x),min(h,d-y)
  if c<=a or d<=b:continue
  if flip:a,c=w-c,w-a
  result.append([a,b,c,d])
 return result
def render(raw,seed,profile,flip):
 rng=np.random.default_rng(seed);ex=float(rng.uniform(.85,1.12));gamma=float(rng.uniform(.92,1.08))
 if profile==0:ex=float(rng.uniform(.85,.95));gamma=float(rng.uniform(1.,1.08))
 if profile==1:ex=float(rng.uniform(1.02,1.12));gamma=float(rng.uniform(.92,1.))
 wb=np.array([rng.uniform(.92,1.08),1.,rng.uniform(.92,1.08)],np.float32)
 saturation=float(rng.uniform(.8,1.15));sigma=float(rng.uniform(.4,1.5));gx,gy=rng.uniform(-.1,.1,2)
 a=np.power(raw.astype(np.float32)/255.,gamma)*255.*ex*wb
 h,w=raw.shape[:2];lighting=1+gx*np.linspace(-1,1,w,dtype=np.float32)[None,:]+gy*np.linspace(-1,1,h,dtype=np.float32)[:,None]
 a=np.clip(a*lighting[:,:,None],0,255).astype(np.uint8)
 hsv=cv2.cvtColor(a,cv2.COLOR_BGR2HSV);hsv[:,:,1]=np.clip(hsv[:,:,1].astype(np.float32)*saturation,0,255).astype(np.uint8)
 a=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR).astype(np.float32);a+=rng.normal(0,sigma,a.shape).astype(np.float32)
 if flip:a=a[:,::-1]
 return np.clip(a,0,255).astype(np.uint8),dict(seed=seed,exposure=ex,gamma=gamma,wb_bgr=wb.tolist(),saturation=saturation,noise_sigma=sigma,lighting_xy=[float(gx),float(gy)],horizontal_flip=flip)
def fits_resize(mask,w,h):
 scale=1600/max(w,h);nw,nh=int(w*scale+.5),int(h*scale+.5)
 return bool(cv2.resize(mask,(nw,nh),interpolation=cv2.INTER_NEAREST).any())
def choose_window(im,anns,index,rng):
 w,h=im['width'],im['height'];target=[j for j,a in enumerate(anns) if a['category_id'] in TARGETS]
 sizes=list(dict.fromkeys([ [800,1024,1200,1024][index],1024,1200,1400,1600,min(w,h)]))
 sizes=[s for s in sizes if s<=min(w,h)]
 available=sorted({anns[j]['category_id'] for j in target});focus=available[(index+im['id'])%len(available)] if available else None
 candidates=[j for j in target if anns[j]['category_id']==focus]+[j for j in target if anns[j]['category_id']!=focus]
 if not candidates:candidates=[None]
 bounds=np.array([[a['bbox'][0],a['bbox'][1],a['bbox'][0]+a['bbox'][2],a['bbox'][1]+a['bbox'][3]] for a in anns if a['category_id'] in TARGETS],dtype=np.float32).reshape(-1,4)
 for size in sizes:
  for chosen in candidates:
   if chosen is not None:
    xx,yy,ww,hh=anns[chosen]['bbox'];cx,cy=xx+ww/2,yy+hh/2
   for attempt in range(100):
    if chosen is None:x,y=rng.randint(0,w-size),rng.randint(0,h-size)
    else:x,y=int(np.clip(cx-rng.uniform(.2,.8)*size,0,w-size)),int(np.clip(cy-rng.uniform(.2,.8)*size,0,h-size))
    if len(bounds):
     intersects=(bounds[:,0]<x+size)&(bounds[:,2]>x)&(bounds[:,1]<y+size)&(bounds[:,3]>y)
     inside=(bounds[:,0]>=x)&(bounds[:,2]<=x+size)&(bounds[:,1]>=y)&(bounds[:,3]<=y+size)
     if (intersects&~inside).any():continue
     if chosen is not None and not(x<=cx<x+size and y<=cy<y+size):continue
    return (x,y,size,size),focus,False
 return (0,0,w,h),focus,True

