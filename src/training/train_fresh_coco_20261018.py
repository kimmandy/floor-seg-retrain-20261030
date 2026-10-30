"""Two-stage NX training. Explicit partial ROI supervision; no automatic deployment."""
import argparse,collections,copy,hashlib,json,math,os,random,shutil,signal,time
from pathlib import Path
from types import MethodType
from concurrent.futures import ThreadPoolExecutor
import training_core_20261017 as core

ROOT=Path('/home/user-nx/kmj/training')
STOP=False

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def emit(path,value):core.write_json(Path(path),value)
def read_data(path):
 d=json.loads(Path(path).read_text());aa=collections.defaultdict(list)
 for a in d['annotations']:aa[a['image_id']].append(a)
 return d,aa

def coverage(anchors,labels,weights,rectangles,unknown,classes):
 import torch
 positive=(labels>=0)&(labels<classes)
 known=torch.ones_like(labels,dtype=torch.bool) if rectangles is None else torch.zeros_like(labels,dtype=torch.bool)
 for x,y,X,Y in rectangles or []:known|=(anchors[:,0]>=x)&(anchors[:,0]<X)&(anchors[:,1]>=y)&(anchors[:,1]<Y)
 for x,y,X,Y in unknown or []:known&=~((anchors[:,0]>=x)&(anchors[:,0]<X)&(anchors[:,1]>=y)&(anchors[:,1]<Y))
 return weights*(known|positive).to(weights.dtype)

def patch_partial(head):
 original=head._get_targets_single
 def wrapped(self,cls_scores,bbox_preds,flat_anchors,valid_flags,gt_instances,img_meta,gt_instances_ignore=None,unmap_outputs=True):
  a,l,w,b,m,s=original(cls_scores,bbox_preds,flat_anchors,valid_flags,gt_instances,img_meta,gt_instances_ignore,unmap_outputs=unmap_outputs)
  w=coverage(a,l,w,img_meta.get('supervision_boxes_input'),img_meta.get('uncertain_boxes_input'),self.num_classes)
  return a,l,w,b,m,s
 head._get_targets_single=MethodType(wrapped,head)

def pipeline(side,train):
 from mmengine.dataset import Compose
 p=[dict(type='LoadImageFromFile')]
 if train:p.append(dict(type='LoadAnnotations',with_bbox=True,with_mask=True,poly2mask=True))
 p += [dict(type='Resize',scale=(side,side),keep_ratio=True),dict(type='Pad',size_divisor=32,pad_val=dict(img=(114,114,114))),
       dict(type='PackDetInputs',meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor'))]
 return Compose(p)

def load_batch(images,aa,pipe):
 from mmengine.dataset import pseudo_collate
 samples=[]
 for im in images:
  instances=[]
  for a in aa[im['id']]:
   x,y,w,h=a['bbox'];instances.append(dict(bbox=[x,y,x+w,y+h],bbox_label=a['category_id']-1,mask=a['segmentation'],ignore_flag=0))
  s=pipe(dict(img_id=im['id'],img_path=im['file_name'],height=im['height'],width=im['width'],instances=instances))
  sx,sy=s['data_samples'].metainfo['scale_factor'];meta={}
  for src,dst in [('supervision_boxes_xyxy','supervision_boxes_input'),('uncertain_boxes_xyxy','uncertain_boxes_input')]:
   if im.get(src) is not None:meta[dst]=[[x*sx,y*sy,X*sx,Y*sy]for x,y,X,Y in im[src]]
  s['data_samples'].set_metainfo(meta);samples.append(s)
 return pseudo_collate(samples)

def build(cfg,classes,checkpoint,out,tag):
 import torch
 from mmengine.config import Config
 from mmdet.registry import MODELS
 from bounded_inference_crop_20261017 import install_bounded_mask_inference
 c=Config.fromfile(cfg['model_config']);c.model.bbox_head.num_classes=len(classes)
 c.model.bbox_head.mask_loss_stride=2;c.model.backbone.init_cfg=None
 for tc in [c.model.test_cfg,c.model.bbox_head.test_cfg]:tc.score_thr=.05;tc.max_per_img=100
 model=MODELS.build(c.model);model.init_weights()
 ck=torch.load(checkpoint,map_location='cpu',weights_only=False);source=ck['state_dict'];target=model.state_dict();adapted=[]
 assert ck['meta']['dataset_meta']['classes'] in [('spill','fastener'),['spill','fastener'],tuple(classes),classes]
 assert set(source)==set(target)
 for key,weight in source.items():
  if weight.shape==target[key].shape:target[key]=weight
  elif 'bbox_head.rtm_cls.'in key and weight.shape[0]==2 and target[key].shape[0]==3:
   target[key][:2]=weight;assert torch.equal(target[key][:2],weight);adapted.append(key)
  else:raise ValueError((key,weight.shape,target[key].shape))
 model.load_state_dict(target,strict=True);model.dataset_meta={'classes':tuple(classes)}
 c.metainfo=model.dataset_meta;c.test_pipeline=pipeline_config(cfg['side'])
 c.dump(str(out/f'{tag}_model_config.py'))
 emit(out/f'{tag}_initialization.json',dict(checkpoint=str(checkpoint),sha256=digest(checkpoint),source_epoch=ck.get('epoch'),
     source_classes=ck['meta']['dataset_meta']['classes'],classes=classes,adapted_classifier_rows=adapted,optimizer_reset=True))
 del ck,source,target
 model.cuda();patch_partial(model.bbox_head);core.install_chunked_mask_loss(model.bbox_head,cfg['mask_loss_chunk'])
 install_bounded_mask_inference(model.bbox_head,4);core.train_mode(model)
 return model

def pipeline_config(side):
 return [dict(type='LoadImageFromFile'),dict(type='Resize',scale=(side,side),keep_ratio=True),dict(type='Pad',size_divisor=32,pad_val=dict(img=(114,114,114))),dict(type='PackDetInputs',meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor'))]

def optimizer(model,cfg,stage):
 import torch
 groups=collections.defaultdict(list)
 for n,p in model.named_parameters():
  factor=cfg['classifier_lr_multiplier'] if stage=='current'and 'bbox_head.rtm_cls.'in n else 1.
  decay=0. if p.ndim==1 or n.endswith('.bias')else cfg['weight_decay']
  groups[(factor,decay)].append(p)
 return torch.optim.AdamW([dict(params=p,lr=cfg['lr'][stage]*f,lr_factor=f,weight_decay=d)for (f,d),p in groups.items()],foreach=True)

def checkpoint(model,opt,scaler,cfg,state,out,name):
 import torch
 def cpu(v):
  if isinstance(v,torch.Tensor):return v.detach().cpu()
  if isinstance(v,dict):return {k:cpu(x)for k,x in v.items()}
  if isinstance(v,list):return [cpu(x)for x in v]
  if isinstance(v,tuple):return tuple(cpu(x)for x in v)
  return v
 saved=dict(state_dict=cpu(model.state_dict()),optimizer=cpu(opt.state_dict()),scaler=scaler.state_dict(),state=state,config=cfg,
            epoch=state['epoch'],meta=dict(dataset_meta=model.dataset_meta,validated_for_deployment=False,selection_scope='ROI diagnostic / training fit; not independent current-camera accuracy'),
            rng_python=random.getstate(),rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state())
 p=out/name;temp=p.with_suffix('.tmp');torch.save(saved,temp);temp.replace(p)

def rate(cfg,stage,local_epoch,update):
 horizon=cfg['schedule_epochs'][stage];progress=min(1.,local_epoch/horizon)
 return cfg['lr'][stage]*(.1+.9*(1+math.cos(math.pi*progress))/2)*min(1.,(update+1)/cfg['warmup_updates'])

def resource_guard(cfg,out):
 if STOP:raise InterruptedError('User/service stop requested')
 if shutil.disk_usage(out).free<cfg['disk_stop_gib']*1024**3:raise RuntimeError('Disk free space below configured reserve')
 return core.guard(cfg['safety'],out)

def metrics(tp,fp,fn):
 return dict(tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp)if tp+fp else None,recall=tp/(tp+fn)if tp+fn else None,
             f1=2*tp/(2*tp+fp+fn)if 2*tp+fp+fn else None,instance_match_score=tp/(tp+fp+fn)if tp+fp+fn else None,
             false_discovery_rate=fp/(tp+fp)if tp+fp else None,miss_rate=fn/(tp+fn)if tp+fn else None)

def eligible(box,im):
 x,y,X,Y=box;cx=(x+X)/2;cy=(y+Y)/2
 rects=im.get('supervision_boxes_xyxy')
 known=rects is None or any(a<=cx<c and b<=cy<d for a,b,c,d in rects)
 return known and not any(a<=cx<c and b<=cy<d for a,b,c,d in im.get('uncertain_boxes_xyxy',[]))

def evaluate(model,cfg,out,epoch,dataset,classes,name,keep_images=False):
 import torch,cv2,numpy as np
 from pycocotools import mask as mu
 d,aa=dataset;pipe=pipeline(cfg['side'],False);model.eval();counts={c:[0,0,0]for c in classes};tiles=[];per_image=[]
 for im in d['images']:
  resource_guard(cfg,out)
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):pred=model.test_step(load_batch([im],aa,pipe))[0].pred_instances.cpu()
  pred=pred[pred.scores>=cfg['score_threshold']];encoded=[]
  for mask in pred.masks.numpy():encoded.append(mu.encode(np.asfortranarray(mask.astype(np.uint8))))
  row=dict(source=im.get('source_key'),counts={})
  for cid,c in enumerate(classes,1):
   gt=[a['segmentation']for a in aa[im['id']]if a['category_id']==cid]
   indices=[j for j in range(len(pred))if int(pred.labels[j])==cid-1];matched=set();used=set();tp=0
   if indices and gt:
    ious=mu.iou([encoded[j]for j in indices],gt,[0]*len(gt))
    for k,j in sorted(enumerate(indices),key=lambda z:-float(pred.scores[z[1]])):
     options=[(ious[k,g],g)for g in range(len(gt))if g not in matched]
     if options:
      score,g=max(options)
      if score>=cfg['mask_iou_threshold']:tp+=1;matched.add(g);used.add(j)
   fp=sum(j not in used and eligible(pred.bboxes[j].tolist(),im)for j in indices);fn=len(gt)-tp
   counts[c][0]+=tp;counts[c][1]+=fp;counts[c][2]+=fn;row['counts'][c]=metrics(tp,fp,fn)
  per_image.append(row)
  if keep_images and len(tiles)<4:
   raw=cv2.imread(im['file_name']);left=raw.copy();right=raw.copy();colors={1:(255,255,0),2:(255,0,255),3:(0,160,255)}
   for a in aa[im['id']]:
    contours,_=cv2.findContours(mu.decode(a['segmentation']),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(left,contours,-1,colors[a['category_id']],max(1,im['width']//700))
   for j in range(len(pred)):
    contours,_=cv2.findContours(pred.masks[j].numpy().astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(right,contours,-1,colors[int(pred.labels[j])+1],max(1,im['width']//700))
   pair=np.concatenate([cv2.resize(left,(640,400)),cv2.resize(right,(640,400))],axis=1)
   cv2.putText(pair,f'{im.get("source_key")} labels / prediction | {name}',(10,20),0,.5,(255,255,255),1);tiles.append(pair)
  del pred,encoded
 per_class={c:metrics(*v)for c,v in counts.items()};f1=[v['f1']for v in per_class.values()if v['f1']is not None]
 result=dict(epoch=epoch,scope=name,images=len(d['images']),classes=per_class,macro_f1=sum(f1)/len(f1)if f1 else None,
   score_threshold=cfg['score_threshold'],mask_iou_threshold=cfg['mask_iou_threshold'],independent_current_evaluation=False,
   unmatched_prediction_policy='count FP only when predicted box centre is in known ROI and outside uncertain ROI',per_image=per_image)
 if tiles:
  header=np.zeros((40,1280,3),np.uint8);cv2.putText(header,f'Epoch {epoch} {name} - NOT independent current-camera accuracy',(10,27),0,.6,(255,255,255),1)
  path=out/f'epoch_{epoch:06d}_{name}.jpg';cv2.imwrite(str(path),np.concatenate([header,*tiles],axis=0),[cv2.IMWRITE_JPEG_QUALITY,85]);result['image']=str(path)
 core.train_mode(model);core.release_cache();return result

def fixed_current(data):
 d,aa=data;ids=['f002436','f002593','f001888','f000055','f001979','f002057','f002139','f002218','f002285','f002357','f000410','f001015']
 images=[]
 for key in ids:
  im=next((i for i in d['images']if i['source_key']==key),None)
  if im:images.append(im)
 assert len(images)>=8
 return dict(images=images),aa

def verify_inputs(cfg,recipe,out):
 hashes={}
 for key,item in recipe['datasets'].items():
  assert digest(item['path'])==item['sha256'],key
  hashes[item['path']]=item['sha256']
  d,_=read_data(item['path']);assert [c['name']for c in d['categories']]==(['spill','fastener','reflection']if key=='current_train'else['spill','fastener'])
  for im in d['images']:assert digest(im['file_name'])==im['image_sha256']
 emit(out/'input_verification.json',dict(manifests=hashes,all_images_hash_verified=True,training_data_frozen=True))

def selftest():
 import torch
 a=torch.tensor([[1.,1.],[5.,5.],[9.,9.],[15.,15.]])
 labels=torch.tensor([3,3,1,3]);w=torch.ones(4)
 assert coverage(a,labels,w,[],[],3).tolist()==[0,0,1,0]
 assert coverage(a,labels,w,[[0,0,10,10]],[[4,4,10,10]],3).tolist()==[1,0,1,0]
 assert coverage(a,labels,w,None,[],3).tolist()==[1,1,1,1]
 assert metrics(2,1,1)['f1']==2/3
 assert not eligible([10,10,12,12],dict(supervision_boxes_xyxy=[]))
 assert eligible([0,0,2,2],dict(supervision_boxes_xyxy=[[0,0,3,3]]))

def main():
 import torch,cv2,numpy as np
 from mmdet.utils import register_all_modules
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--benchmark',type=int,default=0);ap.add_argument('--batch',type=int);args=ap.parse_args()
 cfg=json.loads(Path(args.config).read_text());out=Path(cfg['output']);out.mkdir(exist_ok=True)
 if args.batch:cfg['batch_size']=args.batch;cfg['accumulation']=cfg['effective_batch']//args.batch
 def stop(*_):
  global STOP;STOP=True;core.STOP=True
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 torch.set_num_threads(cfg['cpu_threads']);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=False
 torch.cuda.set_per_process_memory_fraction(cfg['cuda_limit_mib']*1024**2/torch.cuda.get_device_properties(0).total_memory)
 random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);register_all_modules(init_default_scope=True);selftest()
 recipe=json.loads(Path(cfg['recipe']).read_text());verify_inputs(cfg,recipe,out)
 data={key:read_data(v['path'])for key,v in recipe['datasets'].items()}
 fit=fixed_current(data['current_train']);classes=['spill','fastener'];state=dict(stage='base',epoch=0,next_batch=0,step=0,updates=0,best=-1.,bad_epochs=0)
 ck=None
 if not args.benchmark and (out/'latest.pth').exists():
  ck=torch.load(out/'latest.pth',map_location='cpu',weights_only=False);state=ck['state']
  assert ck['config']==cfg,'Resume configuration changed; explicit migration required'
 if args.benchmark:state['stage']='current'
 if state['stage']=='current':classes+=['reflection']
 model=build(cfg,classes,out/'latest.pth'if ck else cfg['checkpoint'],out,'benchmark'if args.benchmark else state['stage'])
 opt=optimizer(model,cfg,state['stage']);scaler=torch.amp.GradScaler('cuda',init_scale=cfg['amp_initial_scale'])
 if ck:
  opt.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler']);random.setstate(ck['rng_python']);torch.set_rng_state(ck['rng_torch']);torch.cuda.set_rng_state(ck['rng_cuda']);del ck
 pipe=pipeline(cfg['side'],True);timings=[];opt.zero_grad(set_to_none=True)
 if args.benchmark:
  d,aa=data['current_train'];base,ba=data['base_train']
  # Actual dense masks, old/public positives, and reflection-only inputs exercise both class mappings.
  representatives=[max(d['images'],key=lambda i:len(aa[i['id']])),next(i for i in d['images']if i.get('scene_kind')=='mixed'),next(i for i in d['images']if i.get('scene_kind')=='reflection_only')]
  sources=[(i,aa)for i in representatives]+[(max(base['images'],key=lambda i:len(ba[i['id']])),ba)]
  for step in range(args.benchmark):
   resource_guard(cfg,out);im,lookup=sources[step%len(sources)];t=time.monotonic()
   batch=model.data_preprocessor(load_batch([im]*cfg['batch_size'],lookup,pipe),training=True)
   with torch.autocast('cuda',dtype=torch.float16):loss,parts=model.parse_losses(model(**batch,mode='loss'))
   assert torch.isfinite(loss);scaler.scale(loss/cfg['accumulation']).backward()
   if (step+1)%cfg['accumulation']==0:
    scaler.unscale_(opt);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.);assert torch.isfinite(norm);scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
   torch.cuda.synchronize();timings.append(time.monotonic()-t);print('BENCH',step,round(timings[-1],3),float(loss.detach()),flush=True);del batch,loss,parts
  opt.zero_grad(set_to_none=True)
  diag=evaluate(model,cfg,out,-cfg['batch_size'],(dict(images=fit[0]['images'][:1]),fit[1]),classes,'smoke_fit',True)
  state['step']=args.benchmark;checkpoint(model,opt,scaler,cfg,state,out,f'smoke_batch{cfg["batch_size"]}.pth')
  loaded=torch.load(out/f'smoke_batch{cfg["batch_size"]}.pth',map_location='cpu',weights_only=False);assert loaded['meta']['dataset_meta']['classes']==tuple(classes)
  model.load_state_dict(loaded['state_dict']);opt.load_state_dict(loaded['optimizer']);scaler.load_state_dict(loaded['scaler'])
  assert scaler.get_scale()==loaded['scaler']['scale']
  emit(out/f'benchmark_batch{cfg["batch_size"]}.json',dict(batch=cfg['batch_size'],accumulation=cfg['accumulation'],steps=args.benchmark,seconds=timings,
       steady_seconds_per_image=sum(timings[2:])/len(timings[2:])/cfg['batch_size'],peak_cuda_mib=torch.cuda.max_memory_allocated()/1024**2,resources=core.snapshot(),checkpoint_reload_verified=True,optimizer_scaler_restore_verified=True,three_class_inference_verified=True))
  return
 emit(out/'effective_config.json',cfg)
 for epoch in range(state['epoch'],cfg['max_epochs']):
  if epoch==cfg['base_epochs'] and state['stage']=='base':
   del model,opt,scaler;core.release_cache();state.update(stage='current',best=-1.,bad_epochs=0,updates=0,next_batch=0)
   classes=['spill','fastener','reflection'];model=build(cfg,classes,out/'best_base.pth',out,'current');opt=optimizer(model,cfg,'current');scaler=torch.amp.GradScaler('cuda',init_scale=cfg['amp_initial_scale'])
  stage=state['stage'];local_epoch=epoch if stage=='base'else epoch-cfg['base_epochs'];d,aa=data[stage+'_train']
  if state['best']<0:
   diagnostic=evaluate(model,cfg,out,epoch,data['base_valid'],['spill','fastener'],'legacy_roi_validation',True)
   current=evaluate(model,cfg,out,epoch,fit,classes,'current_training_fit',True)if stage=='current'else None
   state['reference_legacy_f1']=diagnostic['macro_f1'];state['best']=current['macro_f1']if current else diagnostic['macro_f1']
   checkpoint(model,opt,scaler,cfg,state,out,'best_base.pth'if stage=='base'else'best_current_fit.pth')
   emit(out/f'baseline_{stage}.json',dict(legacy=diagnostic,current_fit=current))
  weights=[recipe['sample_weights'][stage][i['source_key']]for i in d['images']]
  order=random.Random(cfg['seed']+epoch).choices(d['images'],weights=weights,k=len(d['images']))
  batches=[order[i:i+cfg['batch_size']]for i in range(0,len(order),cfg['batch_size'])]
  emit(out/'sampling_latest.json',dict(epoch=epoch+1,stage=stage,source_keys=[i['source_key']for i in order],replacement=True))
  epoch_start=time.monotonic();loss_sum=0.;nloss=0;start_batch=state['next_batch'];opt.zero_grad(set_to_none=True)
  with ThreadPoolExecutor(max_workers=1)as pool:
   future=pool.submit(load_batch,batches[start_batch],aa,pipe)if start_batch<len(batches)else None
   for index in range(start_batch,len(batches)):
    resource_guard(cfg,out);tic=time.monotonic();loaded=future.result()
    future=pool.submit(load_batch,batches[index+1],aa,pipe)if index+1<len(batches)else None
    batch=model.data_preprocessor(loaded,training=True);lr=rate(cfg,stage,local_epoch+index/len(batches),state['updates'])
    for group in opt.param_groups:group['lr']=lr*group['lr_factor']
    with torch.autocast('cuda',dtype=torch.float16):loss,parts=model.parse_losses(model(**batch,mode='loss'))
    if not torch.isfinite(loss):
     retry=dict(epoch=epoch+1,batch=index+1,source_keys=[im['source_key']for im in batches[index]],
                fp16_losses={k:str(float(v.detach()))for k,v in parts.items()},at=time.time())
     # The mask head mutates GT masks in-place. Reload this exact batch before retrying.
     # Release the failed graph; keep any accumulated gradients from earlier microbatches.
     del loss,parts,batch,loaded
     loaded=load_batch(batches[index],aa,pipe);batch=model.data_preprocessor(loaded,training=True)
     with torch.autocast('cuda',enabled=False):loss,parts=model.parse_losses(model(**batch,mode='loss'))
     retry.update(fp32_finite=bool(torch.isfinite(loss)),fp32_losses={k:str(float(v.detach()))for k,v in parts.items()})
     with (out/'precision_retries.jsonl').open('a')as f:f.write(json.dumps(retry)+'\n')
     print('PRECISION_RETRY',json.dumps(retry),flush=True)
     if not retry['fp32_finite']:raise RuntimeError('Non-finite loss persists in FP32; no sample skipped')
    window=min(cfg['accumulation'],len(batches)-(index//cfg['accumulation'])*cfg['accumulation'])
    scaler.scale(loss/window).backward();boundary=(index+1)%cfg['accumulation']==0 or index+1==len(batches)
    if boundary:
     scaler.unscale_(opt);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
     finite=bool(torch.isfinite(norm))
     scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
     state['consecutive_overflows']=0 if finite else state.get('consecutive_overflows',0)+1
     if state['consecutive_overflows']>=6:raise RuntimeError('Repeated AMP gradient overflow after scale backoff')
     if finite:state['updates']+=1
    torch.cuda.synchronize();seconds=time.monotonic()-tic;timings.append(seconds);timings=timings[-100:]
    state.update(step=state['step']+1,epoch=epoch,next_batch=index+1);loss_sum+=float(loss.detach());nloss+=1
    if state['step']%cfg['log_interval']==0 or index==start_batch:
     rec=dict(status='training',stage=stage,epoch=epoch+1,batch=index+1,batches=len(batches),step=state['step'],loss=float(loss.detach()),
       losses={k:float(v)for k,v in parts.items()},lr=lr,classifier_lr=lr*(cfg['classifier_lr_multiplier']if stage=='current'else 1),
       seconds=seconds,estimated_epoch_training_seconds=sum(timings)/len(timings)*len(batches),peak_cuda_mib=torch.cuda.max_memory_allocated()/1024**2,resources=core.snapshot())
     emit(out/'status.json',rec)
     with (out/'step_metrics.jsonl').open('a')as f:f.write(json.dumps(rec)+'\n')
     print('TRAIN',json.dumps(rec),flush=True)
    del loaded,batch,loss,parts
    if boundary and state['step']%cfg['checkpoint_batches']==0:checkpoint(model,opt,scaler,cfg,state,out,'latest.pth')
  train_seconds=time.monotonic()-epoch_start;checkpoint(model,opt,scaler,cfg,state,out,'latest.pth')
  emit(out/'status.json',dict(status='evaluating',stage=stage,epoch=epoch+1,step=state['step']))
  legacy=evaluate(model,cfg,out,epoch+1,data['base_valid'],['spill','fastener'],'legacy_roi_validation',True)
  current=evaluate(model,cfg,out,epoch+1,fit,classes,'current_training_fit',True)if stage=='current'else None
  score=current['macro_f1']if current else legacy['macro_f1'];improved=score>state['best']
  if improved:state['best']=score
  drop=state['reference_legacy_f1']-legacy['macro_f1'];state['bad_epochs']=state['bad_epochs']+1 if drop>cfg['quality_drop_absolute']else 0
  state.update(epoch=epoch+1,next_batch=0)
  rec=dict(epoch=epoch+1,stage=stage,step=state['step'],mean_loss=loss_sum/nloss if nloss else None,training_seconds=train_seconds,
           total_seconds=time.monotonic()-epoch_start,legacy=legacy,current_fit=current,selection_score=score,best_score=state['best'],
           selection_scope='current training fit only'if current else'legacy partial ROI validation',independent_current_accuracy_available=False,
           legacy_f1_drop=drop,consecutive_degraded_epochs=state['bad_epochs'],resources=core.snapshot())
  emit(out/f'epoch_{epoch+1:06d}.json',rec)
  with (out/'epoch_metrics.jsonl').open('a')as f:f.write(json.dumps(rec)+'\n')
  checkpoint(model,opt,scaler,cfg,state,out,'latest.pth')
  if improved:checkpoint(model,opt,scaler,cfg,state,out,'best_base.pth'if stage=='base'else'best_current_fit.pth')
  # Evaluate before publishing the rolling checkpoint: the plateau watcher waits for it.
  from distance_line_interval_20261018 import after_epoch
  range_status=after_epoch(epoch+1,out,lambda name:checkpoint(model,opt,scaler,cfg,state,out,name),core.release_cache)
  emit(out/'range_evaluation_status.json',range_status)
  checkpoint(model,opt,scaler,cfg,state,out,f'epoch_recent_{(epoch+1)%3}.pth')
  emit(out/'status.json',dict(status='epoch_complete',**{k:v for k,v in rec.items()if k not in ['legacy','current_fit']}));print('EPOCH',json.dumps(rec),flush=True)
  if local_epoch+1>=cfg['quality_min_epochs'] and state['bad_epochs']>=cfg['quality_patience']:
   emit(out/'status.json',dict(status='quality_stopped',reason='Legacy ROI regression sustained; manual diagnosis required',epoch=epoch+1,legacy_f1_drop=drop))
   emit(out/'training_finished.json',dict(status='quality_stopped',epoch=epoch+1,legacy_f1_drop=drop,finished_at=time.time()));return
 emit(out/'status.json',dict(status='completed',epochs=cfg['max_epochs']))
 emit(out/'training_finished.json',dict(status='completed',epochs=cfg['max_epochs'],finished_at=time.time()))

if __name__=='__main__':
 try:main()
 except BaseException as exc:
  import sys,traceback
  traceback.print_exc()
  try:
   cfg=json.loads(Path(sys.argv[sys.argv.index('--config')+1]).read_text());out=Path(cfg['output'])
   emit(out/'failure.json',dict(error=repr(exc),at=time.time(),resources=core.snapshot()))
   emit(out/'status.json',dict(status='stopped',error=repr(exc),at=time.time()))
  except Exception:pass
  raise
