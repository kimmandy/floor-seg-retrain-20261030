"""Native CUDA fine-tuning with partial-label coverage and bounded mask-loss memory.

All options live in training.yaml. Source datasets/checkpoints remain read-only.
"""
import argparse
import collections
import contextlib
import copy
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import signal
import time
from types import MethodType

os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:64')
import yaml

ROOT = Path(__file__).resolve().parent
STOP = False


def release_cache():
    import ctypes
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    ctypes.CDLL('libc.so.6').malloc_trim(0)


def snapshot():
    mem = dict((l.split(':')[0], int(l.split()[1])) for l in Path('/proc/meminfo').read_text().splitlines())
    temps = {}
    for p in Path('/sys/class/thermal').glob('thermal_zone*'):
        try:
            name = (p/'type').read_text().strip()
            if name in ('cpu-thermal', 'gpu-thermal', 'tj-thermal', 'soc0-thermal', 'soc1-thermal', 'soc2-thermal'):
                temps[name] = int((p/'temp').read_text()) / 1000
        except (OSError, ValueError, TypeError):
            pass
    if not temps:  # 2026-10-29 kmj: 집 PC — 젯슨 센서 대신 NVIDIA GPU 온도
        import subprocess
        try:
            out = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
                                 capture_output=True, text=True, timeout=10).stdout.split()
            temps = {f'nvidia_gpu{i}': float(v) for i, v in enumerate(out)}
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return dict(available_mib=mem['MemAvailable']/1024, temperature_c=max(temps.values()) if temps else None,
                sensors_c=temps)


def write_json(p, value):
    temp = p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(p)


def guard(cfg, out):
    started = time.monotonic()
    waiting = False
    while True:
        if STOP:
            raise InterruptedError('Stop requested; retaining latest durable checkpoint')
        s = snapshot()
        if s['temperature_c'] is None:
            raise RuntimeError('No readable Jetson thermal sensor; refusing unmonitored training')
        if s['temperature_c'] >= cfg['temperature_abort_c']:
            raise RuntimeError(f'Resource emergency: {s}')
        if s['available_mib'] < cfg['available_memory_abort_mib']:
            # Reclaim inactive allocator/host caches before treating low available RAM as persistent.
            # Keep the same temperature and hard memory limits after reclaim.
            before = s
            release_cache()
            s = snapshot()
            print('SAFETY_RECLAIM', json.dumps(dict(before=before, after=s)), flush=True)
            if (s['temperature_c'] is None or s['temperature_c'] >= cfg['temperature_abort_c']
                    or s['available_mib'] < cfg['available_memory_abort_mib']):
                raise RuntimeError(f'Resource emergency after cache reclaim: {s}')
        low = cfg['available_memory_resume_mib'] if waiting else cfg['available_memory_pause_mib']
        hot = cfg['temperature_resume_c'] if waiting else cfg['temperature_pause_c']
        if s['available_mib'] >= low and s['temperature_c'] < hot:
            return s
        if not waiting:
            release_cache()
            s = snapshot()
            if s['available_mib'] >= cfg['available_memory_resume_mib'] and s['temperature_c'] < cfg['temperature_pause_c']:
                return s
        waiting = True
        write_json(out/'safety_status.json', dict(status='cooling_or_memory_wait', **s))
        print('SAFETY_WAIT', json.dumps(s), flush=True)
        if time.monotonic()-started > cfg['max_pause_seconds']:
            raise RuntimeError('Resource wait timed out; stop without consuming more memory')
        time.sleep(cfg['cooldown_poll_seconds'])


def coverage_weights(anchors, labels, weights, rectangles, classes):
    import torch
    covered = torch.zeros_like(labels, dtype=torch.bool)
    for x1,y1,x2,y2 in rectangles:
        covered |= (anchors[:,0]>=x1) & (anchors[:,0]<x2) & (anchors[:,1]>=y1) & (anchors[:,1]<y2)
    positive = (labels >= 0) & (labels < classes)
    return weights * (covered | positive).to(weights.dtype)


def install_partial(head):
    original = head._get_targets_single
    def wrapped(self, cls_scores, bbox_preds, flat_anchors, valid_flags, gt_instances, img_meta,
                gt_instances_ignore=None, unmap_outputs=True):
        a,l,w,b,m,s = original(cls_scores,bbox_preds,flat_anchors,valid_flags,gt_instances,img_meta,
                              gt_instances_ignore,unmap_outputs=unmap_outputs)
        rects = img_meta.get('supervision_boxes_input')
        if rects is not None:
            w = coverage_weights(a,l,w,rects,self.num_classes)
        return a,l,w,b,m,s
    head._get_targets_single = MethodType(wrapped, head)


def install_amp_nms(head):
    # Installed MMCV CUDA NMS requires float32 even when the neural network uses AMP.
    original=head._bbox_mask_post_process
    def wrapped(self,results,*args,**kwargs):
        results.bboxes=results.bboxes.float()
        results.scores=results.scores.float()
        return original(results,*args,**kwargs)
    head._bbox_mask_post_process=MethodType(wrapped,head)


def install_chunked_mask_loss(head, chunk):
    """Same Dice objective/normalizer, checkpoint chunks, never duplicate full-size GT per anchor."""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    def loss(self, mask_feats, flatten_kernels, sampling_results_list, batch_gt_instances):
        count = sum(len(s.pos_inds) for s in sampling_results_list)
        if count == 0:
            return mask_feats.sum()*0
        total = mask_feats.sum()*0
        stride = self.mask_loss_stride
        scale = self.prior_generator.strides[0][0] // stride
        for feat,kernels,s,gt in zip(mask_feats, flatten_kernels, sampling_results_list, batch_gt_instances):
            small_gt = gt.masks[:,stride//2::stride,stride//2::stride]
            for start in range(0,len(s.pos_inds),chunk):
                stop = start+chunk
                kk = kernels[s.pos_inds[start:stop]]
                pp = s.pos_priors[start:stop]
                target = small_gt[s.pos_assigned_gt_inds[start:stop]]
                def part(f,k,p,t):
                    logits = self._mask_predict_by_feat_single(f,k,p)
                    logits = F.interpolate(logits.unsqueeze(0),scale_factor=scale,
                                           mode='bilinear',align_corners=False).squeeze(0)
                    return self.loss_mask(logits,t,weight=None,avg_factor=max(1,count))
                total = total + checkpoint(part, feat, kk, pp, target, use_reentrant=False)
        return total
    head.loss_mask_by_feat = MethodType(loss,head)


def load_manifest(name, config):
    p = ROOT / config['data']['directory'] / f'{name}.json'
    d = json.loads(p.read_text())
    lookup = collections.defaultdict(list)
    for a in d['annotations']:
        x,y,w,h = a['bbox']
        lookup[a['image_id']].append(dict(bbox=[x,y,x+w,y+h],bbox_label=a['category_id']-1,
                                         mask=a['segmentation'],ignore_flag=0))
    return d,lookup


def pipeline(config, train):
    from mmengine.dataset import Compose
    side = config['model']['input_long_side']
    parts = [dict(type='LoadImageFromFile')]
    if train:
        parts.append(dict(type='LoadAnnotations',with_bbox=True,with_mask=True,poly2mask=True))
    parts.append(dict(type='Resize',scale=(side,side),keep_ratio=True))
    if train:
        parts.append(dict(type='YOLOXHSVRandomAug',hue_delta=2,saturation_delta=10,value_delta=10))
    parts += [dict(type='Pad',size_divisor=32,pad_val=dict(img=(114,114,114))),
              dict(type='PackDetInputs',meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor'))]
    return Compose(parts)


def sample(im, lookup, pipe):
    from mmengine.dataset import pseudo_collate
    s = pipe(dict(img_id=im['id'],img_path=im['file_name'],height=im['height'],width=im['width'],
                  instances=lookup[im['id']]))
    if im.get('supervision_boxes_xyxy'):
        sx,sy = s['data_samples'].metainfo['scale_factor']
        rects = [[x1*sx,y1*sy,x2*sx,y2*sy] for x1,y1,x2,y2 in im['supervision_boxes_xyxy']]
        s['data_samples'].set_metainfo(dict(supervision_boxes_input=rects))
    return pseudo_collate([s])


def build_model(config, out):
    import torch
    from mmengine.config import Config
    from mmdet.registry import MODELS
    cfg = Config.fromfile(config['model']['config'])
    cfg.model.bbox_head.num_classes = 2
    cfg.model.bbox_head.mask_loss_stride = config['model']['mask_loss_stride']
    cfg.model.backbone.init_cfg = None
    for test_cfg in [cfg.model.test_cfg, cfg.model.bbox_head.test_cfg]:
        test_cfg.score_thr = 0.05; test_cfg.max_per_img = 100
    model = MODELS.build(cfg.model)
    model.init_weights()
    ckpt = torch.load(config['model']['checkpoint'],map_location='cpu',weights_only=False)
    source = ckpt.get('state_dict',ckpt.get('model',ckpt))
    target = model.state_dict()
    adapted=[]
    for name,weight in source.items():
        if name not in target: raise ValueError(f'Unexpected source weight: {name}')
        if weight.shape == target[name].shape:
            target[name] = weight
        elif 'bbox_head.rtm_cls.' in name and weight.shape[0]==1 and target[name].shape[0]==2:
            # Preserve the existing fastener class (new id=1); spill row retains its initialized prior.
            target[name][1:2] = weight
            adapted.append(name)
        else:
            raise ValueError(f'Unsupported checkpoint shape: {name}, {weight.shape}, {target[name].shape}')
    missing=set(target)-set(source)
    if missing: raise ValueError(f'Uninitialized checkpoint keys: {missing}')
    model.load_state_dict(target,strict=True)
    cfg.metainfo = dict(classes=('spill','fastener'))
    cfg.model.bbox_head.mask_loss_stride=config['model']['mask_loss_stride']
    cfg.test_pipeline=[dict(type='LoadImageFromFile'),dict(type='Resize',scale=(config['model']['input_long_side'],)*2,keep_ratio=True),
        dict(type='Pad',size_divisor=32,pad_val=dict(img=(114,114,114))),
        dict(type='PackDetInputs',meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor'))]
    cfg.dump(str(out/'model_config.py'))
    write_json(out/'initialization.json',dict(checkpoint=config['model']['checkpoint'],
        sha256=hashlib.sha256(Path(config['model']['checkpoint']).read_bytes()).hexdigest(),
        adapted_classifier_rows=adapted,failed_combined_checkpoint_used=False))
    del ckpt,source,target
    model.cuda()
    install_partial(model.bbox_head)
    install_amp_nms(model.bbox_head)
    from bounded_inference import install_bounded_mask_inference
    install_bounded_mask_inference(model.bbox_head,config['model']['mask_inference_chunk'])
    install_chunked_mask_loss(model.bbox_head,config['model']['mask_loss_chunk'])
    return model


def train_mode(model):
    import torch
    model.train()
    for m in model.modules():
        if isinstance(m,torch.nn.modules.batchnorm._BatchNorm): m.eval()


def validate(model,config,out,epoch,split='valid'):
    import torch
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as M
    release_cache()
    dataset,lookup = load_manifest(split,config)
    pipe = pipeline(config,False); model.eval(); rows=[]
    prior=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
    visualized_sources=set()
    for validation_index,im in enumerate(dataset['images']):
        resources=guard(config['safety'],out)
        write_json(out/'status.json',dict(status='validating',step=prior.get('step'),epoch=epoch,validation_index=validation_index,validation_total=len(dataset['images']),split=split,**resources))
        if validation_index%10==0: print('VALIDATION_PROGRESS',json.dumps(dict(epoch=epoch,index=validation_index,total=len(dataset['images']),**resources)),flush=True)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16,enabled=config['training']['amp']):
            pred = model.test_step(sample(im,lookup,pipe))[0].pred_instances.cpu()
        for n in range(len(pred)):
            r=M.encode(np.asfortranarray(pred.masks[n].numpy().astype(np.uint8)));r['counts']=r['counts'].decode()
            rows.append(dict(image_id=im['id'],category_id=int(pred.labels[n])+1,
                             score=float(pred.scores[n]),segmentation=r))
        if split=='valid' and im['source'] not in visualized_sources:
            from validation_preview import save_comparison
            annotations=[a for a in dataset['annotations'] if a['image_id']==im['id']]
            save_comparison(im,annotations,pred,config,out/'validation_images'/f'epoch_{epoch:02d}',epoch,prior.get('step'))
            visualized_sources.add(im['source'])
        del pred
        if (validation_index+1) % config['training'].get('evaluation_cleanup_interval', 1) == 0:
            release_cache()
    gt=COCO(str(ROOT/config['data']['directory']/f'{split}.json'))
    write_json(out/f'{split}_predictions_epoch_{epoch:02d}.json',rows)
    evaluator=None
    scores={'spill':0.,'fastener':0.}
    if rows:
        dt=gt.loadRes(rows)
        evaluator=COCOeval(gt,dt,'segm'); evaluator.evaluate();evaluator.accumulate();evaluator.summarize()
        for k,name in enumerate(('spill','fastener')):
            pp=evaluator.eval['precision'][:,:,k,0,-1]; valid=pp[pp>=0];scores[name]=float(valid.mean()) if valid.size else 0.
    from validation_operating_metrics import summarize_evaluator
    full_detection=summarize_evaluator(evaluator,config['inference']['score_threshold'],('spill','fastener'),len(dataset['images']),dataset['annotations'])
    result=dict(epoch=epoch,split=split,segm_AP=scores,full_detection=full_detection,mean_AP=sum(scores.values())/2,
                images=len(dataset['images']),independent_field_validation=False)
    write_json(out/f'{split}_epoch_{epoch:02d}.json',result)
    print('VALIDATION',json.dumps(result),flush=True)
    train_mode(model);torch.cuda.empty_cache()
    return result


def private_fit_check(model,config,out,epoch):
    """Regression diagnostic on the 100 training captures; explicitly NOT held-out accuracy."""
    import torch
    import numpy as np
    from pycocotools import mask as M
    data,lookup=load_manifest(config['data'].get('fit_diagnostic_split','train'),config)
    images=[i for i in data['images'] if i['training_role']=='original_roi_sampling_copy' and i['source_image_id']%2==1]
    pipe=pipeline(config,False);model.eval();matched=total=spill=oversized=0
    for diagnostic_index,im in enumerate(images):
        guard(config['safety'],out)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16,enabled=config['training']['amp']):
            pred=model.test_step(sample(im,lookup,pipe))[0].pred_instances.cpu()
        pred=pred[pred.scores>=config['inference']['score_threshold']]
        spill+=int((pred.labels==0).sum())
        masks=pred.masks[pred.labels==1].numpy().astype(bool)
        oversized+=sum(int(m.sum()>im['width']*im['height']*.1) for m in masks)
        used=set()
        for a in lookup[im['id']]:
            gt=M.decode(a['mask']).astype(bool);total+=1
            candidates=[(np.logical_and(gt,m).sum()/max(1,np.logical_or(gt,m).sum()),n)
                        for n,m in enumerate(masks) if n not in used]
            if candidates:
                iou,n=max(candidates)
                if iou>=.5:matched+=1;used.add(n)
        del pred,masks
        if (diagnostic_index+1) % config['training'].get('evaluation_cleanup_interval', 1) == 0:
            release_cache()
    result=dict(images=len(images),weak_targets=total,matched=matched,recall=matched/max(1,total),
                spill_on_known_dry_roi=spill,oversized_fastener_masks=oversized,independent_validation=False,
                warning='Reused training captures / weak masks: fit regression diagnostic only')
    write_json(out/f'private_fit_epoch_{epoch:02d}.json',result)
    print('PRIVATE_FIT',json.dumps(result),flush=True);train_mode(model);release_cache()
    return result


def save(model,opt,scaler,out,name,epoch,index,step,best,stale,config):
    import torch
    import numpy as np
    release_cache()
    def cpu(v):
        if isinstance(v,torch.Tensor):return v.detach().cpu()
        if isinstance(v,dict):return {k:cpu(x) for k,x in v.items()}
        if isinstance(v,list):return [cpu(x) for x in v]
        if isinstance(v,tuple):return tuple(cpu(x) for x in v)
        return v
    state=dict(state_dict=cpu(model.state_dict()),optimizer=cpu(opt.state_dict()),scaler=scaler.state_dict(),
               epoch=epoch,next_index=index,step=step,best=best,stale=stale,config=config,
               rng_python=random.getstate(),rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state(),
               rng_numpy=np.random.get_state(),
               meta=dict(dataset_meta={'classes':('spill','fastener')},measured_on='<host>',validated_for_deployment=False))
    temp=out/(name+'.tmp');torch.save(state,temp);temp.replace(out/name)


def learning_rate(tc, epoch, index, epoch_samples, step):
    schedule_epochs = tc.get('lr_schedule_epochs', tc['epochs'])
    if schedule_epochs <= 0 or epoch_samples <= 0:
        raise ValueError('Learning-rate schedule and epoch sample counts must be positive')
    progress = min(1., (epoch + index / epoch_samples) / schedule_epochs)
    return tc['learning_rate'] * (0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2) * min(1., (step + 1) / 100)


def main():
    import torch
    import cv2
    import numpy as np
    from mmdet.utils import register_all_modules
    ap=argparse.ArgumentParser();ap.add_argument('--config',default=str(ROOT/'training.yaml'))
    ap.add_argument('--benchmark',type=int,default=0);ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();config=yaml.safe_load(Path(args.config).read_text())
    tc=config['training']; sc=config['safety']
    from field_review_gate import validate_review
    validate_review(ROOT, required=tc.get('require_full_field_review',False))
    if tc['batch_size']!=1:raise ValueError('This guarded implementation intentionally supports batch=1 only')
    out=ROOT/(f'benchmark_{config["model"]["input_long_side"]}_{time.strftime("%H%M%S")}' if args.benchmark else tc['output'])
    if out.exists() and not args.resume and any(p.is_file() for p in out.rglob('*')):
        raise RuntimeError('Fresh training requires an empty, archived output directory')
    out.mkdir(exist_ok=True)
    write_json(out/'effective_config.json',config)
    def stop(*_):
        global STOP;STOP=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    torch.set_num_threads(tc['cpu_threads']);cv2.setNumThreads(1)
    torch.cuda.set_per_process_memory_fraction(sc['cuda_allocator_limit_mib']*1024**2/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cudnn.benchmark=False
    random.seed(tc['seed']);np.random.seed(tc['seed']);torch.manual_seed(tc['seed'])
    register_all_modules(init_default_scope=True)
    guard(sc,out);model=build_model(config,out);train_mode(model)
    opt=torch.optim.AdamW(model.parameters(),lr=tc['learning_rate'],weight_decay=tc['weight_decay'],
                          foreach=tc.get('optimizer_foreach',False))
    scaler=torch.amp.GradScaler('cuda',enabled=tc['amp'])
    data,lookup=load_manifest('train',config);pipe=pipeline(config,True)
    from field_incremental_intake import range_order
    order,range_sampling=range_order(data['images'],config['data']['repeat'],config['data'].get('field_range_ratio'))
    import hashlib
    train_manifest_sha256=hashlib.sha256((ROOT/config['data']['directory']/'train.json').read_bytes()).hexdigest()
    sampling_record=dict(at=time.time(),train_manifest_sha256=train_manifest_sha256,range_sampling=range_sampling,
                         order_image_ids=[im['id'] for im in order],epochs=tc['epochs'])
    write_json(out/'sampling_audit.json',sampling_record)
    with (out/'sampling_events.jsonl').open('a') as f:f.write(json.dumps(sampling_record)+'\n')
    epoch_start,index_start,step,best,stale=0,0,0,-1.,0
    if args.resume:
        ck=torch.load(out/'latest.pth',map_location='cpu',weights_only=False)
        if ck['config']['training'].get('selection_metric') != tc.get('selection_metric'):
            raise RuntimeError('Checkpoint selection metric differs; migrate best/stale before resuming')
        model.load_state_dict(ck['state_dict']);opt.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler'])
        # Optimizer restore includes the old param_groups execution setting.
        # Explicitly apply the measured execution mode without resetting state.
        for group in opt.param_groups:group['foreach']=tc.get('optimizer_foreach',False)
        epoch_start,index_start,step,best,stale=ck['epoch'],ck['next_index'],ck['step'],ck['best'],ck['stale']
        random.setstate(ck['rng_python']);torch.set_rng_state(ck['rng_torch']);torch.cuda.set_rng_state(ck['rng_cuda'])
        np.random.set_state(ck['rng_numpy'])
        resume_record=dict(checkpoint_epoch=epoch_start,next_index=index_start,step=step,
            configured_epochs=tc['epochs'],checkpoint_configured_epochs=ck['config']['training']['epochs'],
            optimizer_restored=True,scaler_restored=True,rng_restored=['python','numpy','torch','cuda'],train_manifest_sha256=train_manifest_sha256,range_sampling=range_sampling)
        with (out/'resume_events.jsonl').open('a') as f:f.write(json.dumps(resume_record)+'\n')
        print('RESUME',json.dumps(resume_record),flush=True)
        del ck
    if args.benchmark:
        # Include a worst instance-count public frame as well as all normal input groups.
        largest=max((i for i in data['images'] if i['source']=='fastener'),key=lambda i:len(lookup[i['id']]))
        sources=sorted({i['source'] for i in data['images'] if i['source']!='fastener'})
        others=[max((i for i in data['images'] if i['source']==s),key=lambda i:len(lookup[i['id']])) for s in sources]
        extra=[]
        for predicate in (lambda i:not lookup[i['id']], lambda i:i.get('supervision_boxes_xyxy') and lookup[i['id']]):
            candidate=next((i for i in data['images'] if predicate(i)),None)
            if candidate is not None:extra.append(candidate)
        representatives=[largest,*others,*extra]
        order=(representatives*math.ceil(args.benchmark/len(representatives)))[:args.benchmark]
    opt.zero_grad(set_to_none=True)
    start=time.monotonic();timings=[];total_steps=len(order)*(1 if args.benchmark else tc['epochs'])
    print('START',json.dumps(dict(input_side=config['model']['input_long_side'],epoch_samples=len(order),epochs=tc['epochs'],resources=snapshot())),flush=True)
    write_json(out/'status.json',dict(status='training',step=step,total_steps=total_steps,started_unix=time.time(),resources=snapshot()))
    try:
        if not args.benchmark and step==0:
            from result_preview import preview
            preview(model,config,out,'baseline')
            if tc.get('field_quality_gate',False):
                from field_fit_diagnostic import evaluate_field_fit
                write_json(out/'status.json',dict(status='baseline_field_validation',step=0,
                    started_unix=time.time(),resources=snapshot()))
                evaluate_field_fit(model,config,out,0)
        for epoch in range(epoch_start,1 if args.benchmark else tc['epochs']):
            epoch_started=time.monotonic()
            shuffled=order.copy()
            if not args.benchmark:random.Random(tc['seed']+epoch).shuffle(shuffled)
            for index,im in enumerate(shuffled):
                if epoch==epoch_start and index<index_start:continue
                guard(sc,out);tic=time.monotonic()
                lr=learning_rate(tc,epoch,index,len(shuffled),step)
                for group in opt.param_groups:group['lr']=lr
                batch=model.data_preprocessor(sample(im,lookup,pipe),training=True)
                with torch.autocast('cuda',dtype=torch.float16,enabled=tc['amp']):
                    losses=model(**batch,mode='loss');loss,parts=model.parse_losses(losses)
                if not torch.isfinite(loss):raise RuntimeError(f'Non-finite loss at {im["file_name"]}')
                scaler.scale(loss/tc['accumulation']).backward()
                step+=1
                if (index+1)%tc['accumulation']==0 or index+1==len(shuffled):
                    scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
                    scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize();duration=time.monotonic()-tic;timings.append(duration)
                rec=dict(step=step,epoch=epoch+1,index=index+1,image_id=im['id'],source=im['source'],
                    shape=list(batch['inputs'].shape),seconds=duration,loss=float(loss.detach()),
                    losses={k:float(v) for k,v in parts.items()},lr=lr,
                    cuda_peak_mib=torch.cuda.max_memory_allocated()/1024**2,cuda_reserved_mib=torch.cuda.memory_reserved()/1024**2,
                    **snapshot())
                with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
                del batch,losses,loss,parts
                if step%tc['log_interval']==0 or args.benchmark:
                    elapsed=time.monotonic()-start
                    eta=(total_steps-step)*sum(timings[-200:])/len(timings[-200:])
                    write_json(out/'status.json',dict(status='training',total_steps=total_steps,eta_hours=eta/3600,
                        elapsed_seconds=elapsed,**rec))
                    print('TRAIN',json.dumps(rec),flush=True)
                if not args.benchmark and step%tc['save_interval']==0 and (index+1)%tc['accumulation']==0:
                    save(model,opt,scaler,out,'latest.pth',epoch,index+1,step,best,stale,config)
                    if step==config['preview']['first_after_steps']:
                        from result_preview import preview
                        preview(model,config,out,f'step_{step:06d}')
            if args.benchmark:
                # Validation path and resource guard are exercised on a real example below, not the full test set.
                release_cache(); model.eval()
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16,enabled=tc['amp']):
                    predictions=model.test_step(sample(order[1],lookup,pipeline(config,False)))
                del predictions
                from field_fit_diagnostic import evaluate_field_fit
                evaluate_field_fit(model,config,out,0)
                # Exercise checkpoint serialization while still inside the same cgroup limit.
                save(model,opt,scaler,out,'smoke.pth',0,step,step,best,stale,config)
                write_json(out/'benchmark.json',dict(status='passed',steps=step,seconds=timings,
                    mean_seconds=sum(timings)/len(timings),cuda_peak_mib=torch.cuda.max_memory_allocated()/1024**2,
                    resources=snapshot(),worst_source_image=order[0]['file_name']))
                return
            # Persist completed training before validation can fail.
            save(model,opt,scaler,out,'latest.pth',epoch,len(shuffled),step,best,stale,config)
            training_completed=time.monotonic()
            result=validate(model,config,out,epoch+1)
            validation_completed=time.monotonic()
            from field_fit_diagnostic import evaluate_field_fit
            evaluate_field_fit(model,config,out,epoch+1)
            fit=private_fit_check(model,config,out,epoch+1)
            from field_selection import METRIC, selection_score as field_score
            if tc.get('selection_metric') != METRIC:
                raise ValueError('Expected heldout field selection metric')
            heldout=json.loads((out/'heldout_field_validation'/f'epoch_{epoch+1:02d}.json').read_text())
            selection_score=field_score(heldout)
            write_json(out/f'selection_epoch_{epoch+1:02d}.json',dict(selection_score=selection_score,
                metric=METRIC,formula='(heldout spill F1 + heldout fastener F1) / 2',
                heldout_field_validation=heldout,
                public_validation=result,private_training_fit=fit))
            improved=selection_score>best
            if improved:best=selection_score;stale=0
            else:stale+=1
            save(model,opt,scaler,out,'latest.pth',epoch+1,0,step,best,stale,config)
            if improved:save(model,opt,scaler,out,'best.pth',epoch+1,0,step,best,stale,config)
            if tc.get('save_each_epoch',False):save(model,opt,scaler,out,f'epoch{epoch+1:02d}.pth',epoch+1,0,step,best,stale,config)
            if config['preview']['after_each_epoch']:
                from result_preview import preview
                preview(model,config,out,f'epoch_{epoch+1:02d}')
            epoch_timing=dict(epoch=epoch+1,step=step,completed_unix=time.time(),
                training_seconds=training_completed-epoch_started,
                public_validation_seconds=validation_completed-training_completed,
                diagnostics_and_saving_seconds=time.monotonic()-validation_completed,
                total_seconds=time.monotonic()-epoch_started,
                optimizer_foreach=tc.get('optimizer_foreach',False),
                mask_loss_chunk=config['model']['mask_loss_chunk'],
                evaluation_cleanup_interval=tc.get('evaluation_cleanup_interval',1))
            write_json(out/f'timing_epoch_{epoch+1:02d}.json',epoch_timing)
            print('EPOCH_TIMING',json.dumps(epoch_timing),flush=True)
            if tc.get('field_quality_gate',False):
                from field_quality_guard import check_epoch
                quality=check_epoch(ROOT,out,epoch+1)
                print('FIELD_QUALITY_GATE',json.dumps(quality),flush=True)
                if quality['rejected']:
                    write_json(out/'status.json',dict(status='quality_rejected',epoch=epoch+1,
                        step=step,quality=quality,automatic_runtime_deployment=False))
                    return
            if (ROOT/'pending_dataset_update.json').is_file():
                write_json(out/'status.json',dict(status='dataset_update_boundary',epoch=epoch+1,
                    step=step,request=json.loads((ROOT/'pending_dataset_update.json').read_text())))
                return
            if epoch+1>=tc['early_stop_min_epochs'] and stale>=tc['early_stop_patience']:break
        # Final test uses only the selected validation checkpoint, never chooses an epoch from test results.
        ck=torch.load(out/'best.pth',map_location='cpu',weights_only=False);model.load_state_dict(ck['state_dict']);del ck
        test=validate(model,config,out,epoch+1,'test')
        write_json(out/'status.json',dict(status='completed',steps=step,best_selection_score=best,
            test=test,elapsed_seconds=time.monotonic()-start,automatic_runtime_deployment=False))
    except BaseException as exc:
        write_json(out/'status.json',dict(status='stopped',error=repr(exc),step=step,resources=snapshot(),
            latest_checkpoint_exists=(out/'latest.pth').exists()))
        raise


if __name__=='__main__':main()
