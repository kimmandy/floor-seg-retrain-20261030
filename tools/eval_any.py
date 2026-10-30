"""RTMDet-Ins training with explicit stages, preserved partial ROIs, and held-out validation."""
import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path('/home/user/work/setup')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'vendor'))
sys.path.insert(0, str(Path('/home/user/work/setup')))
from prefetch import prefetched_batches
import train_fresh_coco_20261018 as prior
import training_core_20261017 as core
import all6_metrics as evaluation
from periodic_checkpoints import archive_checkpoint
# 2026-10-29 kmj: mmdet 마스크 축소를 중심 정렬(INTER_AREA ≥0.5)로 — 정답 마스크 +0.7 px 밀림 제거
import mask_resize_fix; mask_resize_fix.install()
from distance_training import expand_training_boxes, install_assignment_monitor, install_assignment_coverage, POLICY as TRAINING_POLICY

from all6_sampling import epoch_images, POLICY as CROP_POLICY
import numpy as np
CLASSES = ["water","bolt","nut","oil","paint","reflection"]

def configuration(stage):
    return dict(model_config=str(Path('/home/user/work/setup')/'model_config.py'), side=1600, batch_size=2, accumulation=2,
                lr=1e-4, weight_decay=.01, mask_loss_chunk=16,
                mask_loss_stride=1, amp_dtype='bfloat16', training_policy=TRAINING_POLICY, min_training_box_side=10,
                class_names=list(CLASSES), sampling_policy=CROP_POLICY, selection_policy="macro_f1_six_classes_equal_weight", warmup_batches=500,
                score_threshold=.5, mask_iou_threshold=.5, disk_stop_gib=12,
                safety=dict(available_memory_pause_mib=4096, available_memory_resume_mib=6144,
                            available_memory_abort_mib=2048, temperature_pause_c=83,
                            temperature_resume_c=75, temperature_abort_c=90,
                            max_pause_seconds=900, cooldown_poll_seconds=5))


def build(cfg, stage, checkpoint):
    import torch
    from mmengine import Config
    from mmengine.logging import MMLogger
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    register_all_modules()
    MMLogger.get_instance('bok_training', log_level='ERROR')
    config = Config.fromfile(cfg['model_config'])
    classes = list(CLASSES)
    config.model.bbox_head.num_classes=len(classes)
    config.model.bbox_head.mask_loss_stride=cfg['mask_loss_stride']
    config.model.backbone.init_cfg=None
    model=MODELS.build(config.model)
    model.init_weights()
    if checkpoint:
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        source=ck['state_dict']; target=model.state_dict()
        source_names=list(ck.get('meta',{}).get('dataset_meta',{}).get('classes',[]))
        if len(source_names)==6 and source_names!=CLASSES:raise ValueError('Six-class checkpoint order mismatch')
        for key,value in source.items():
            if key not in target:
                raise ValueError(f'Unexpected checkpoint key: {key}')
            if value.shape==target[key].shape:
                target[key]=value
            elif 'bbox_head.rtm_cls.' in key and value.shape[0]==80:
                pass  # fresh COCO initialization, no target-label rows are inferred
            else:
                raise ValueError(f'Unsupported checkpoint shape: {key}')
        if set(source)!=set(target):
            raise ValueError('Checkpoint architecture mismatch')
        model.load_state_dict(target,strict=True)
    model.dataset_meta=dict(classes=tuple(classes))
    model.cuda()
    prior.patch_partial(model.bbox_head)
    # MMCV CUDA NMS requires matching FP32 boxes/scores under AMP inference.
    core.install_amp_nms(model.bbox_head)
    core.install_chunked_mask_loss(model.bbox_head,cfg['mask_loss_chunk'])
    install_assignment_coverage(model.bbox_head)
    install_assignment_monitor(model.bbox_head)
    # Stock inference preserves resize/pad behavior. AGX has sufficient memory at batch one.
    core.train_mode(model)
    return model,classes


def load_data(dataset,stage,split):
    if split!='train':
        raise ValueError('Use the IMX219-only evaluation loader for valid/test')
    path=dataset/f'{stage}_{split}.coco.json'
    hashes=json.loads((dataset/'manifest_hashes.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==hashes[path.name], 'Dataset manifest changed'
    data=prior.read_data(path)
    assert data[0]['images'], f'{stage}/{split}: no reviewed eligible images yet; see pending_review.json'
    for im in data[0]['images']:
        assert im['eligible'] and im['split']==split
        assert split=='train' or not im['augmented']
        assert Path(im['file_name']).is_file()
        assert hashlib.sha256(Path(im['file_name']).read_bytes()).hexdigest()==im['image_sha256']
    return data


def save(path,model,opt,scaler,epoch,best,stale,cfg,stage):
    import torch
    temporary=path.with_suffix('.tmp')
    torch.save(dict(state_dict=model.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),
                    epoch=epoch,best=best,stale=stale,config=cfg,stage=stage,
                    rng_python=random.getstate(),rng_numpy=np.random.get_state(),rng_torch=torch.get_rng_state(),
                    rng_cuda=torch.cuda.get_rng_state_all(),meta=dict(dataset_meta=model.dataset_meta)),temporary)
    temporary.replace(path)


def training_loss(model, data, amp_dtype):
    """Retry non-finite autocast loss with pristine samples: RTMDet mutates GT masks."""
    import torch
    if amp_dtype != 'bfloat16':
        raise ValueError(f'Unsupported training AMP dtype: {amp_dtype}')
    pristine_samples = copy.deepcopy(data['data_samples'])
    with torch.autocast('cuda', dtype=torch.bfloat16):
        losses = model._run_forward(data, mode='loss')
        loss, _ = model.parse_losses(losses)
    retried = not bool(torch.isfinite(loss))
    if retried:
        # Drop the failed graph before allocating the full-precision retry.
        del loss, losses
        data['data_samples'] = pristine_samples
        with torch.autocast('cuda', enabled=False):
            losses = model._run_forward(data, mode='loss')
            loss, _ = model.parse_losses(losses)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite loss in BF16 and FP32')
    return losses, loss, retried

def optimizer_for(model,cfg):
    import torch
    decay=[];no_decay=[]
    for name,param in model.named_parameters():
        (no_decay if param.ndim==1 or name.endswith('.bias') else decay).append(param)
    return torch.optim.AdamW([dict(params=decay,weight_decay=cfg['weight_decay']),
                             dict(params=no_decay,weight_decay=0.)],lr=cfg['lr'])

def restore(state,model,opt,scaler,cfg):
    import torch
    for k in cfg:
        if state['config'].get(k)!=cfg[k]:raise ValueError('Resume configuration changed: '+k)
    model.load_state_dict(state['state_dict'],strict=True);opt.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
    random.setstate(state['rng_python']);np.random.set_state(state['rng_numpy']);torch.set_rng_state(state['rng_torch']);torch.cuda.set_rng_state_all(state['rng_cuda'])

def run(args):
    import cv2,torch
    torch.set_num_threads(6);cv2.setNumThreads(1);random.seed(20261028);np.random.seed(20261028);torch.manual_seed(20261028)
    if not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA BF16 support required')
    cfg=configuration('current');cfg['early_stopping_patience']=0
    cfg['dataset_manifests']=json.loads((args.dataset/'manifest_hashes.json').read_text())
    cfg['metric_policy']=evaluation.POLICY_VERSION;cfg['schedule_epochs']=args.epochs
    init=json.loads((Path('/home/user/work/setup')/'initialization.json').read_text())
    source=Path(init['checkpoint'])
    if hashlib.sha256(source.read_bytes()).hexdigest()!=init['sha256']: raise ValueError('Initial best checkpoint changed')
    cfg['initialization']=dict(init,kind='previous_valid_best_weights_new_optimizer',new_six_class_head=False)
    if args.command=='train':
        valid=evaluation.load_evaluation(args.dataset,'valid',CLASSES)
        train=load_data(args.dataset,'current','train')
        evaluation.load_evaluation(args.dataset,'test',CLASSES) # membership/hash checks only
        checkpoint=args.resume or source
    else:
        if not args.checkpoint:raise ValueError('Evaluation requires a checkpoint')
        checkpoint=args.checkpoint
        data=evaluation.load_evaluation(args.dataset,args.split,CLASSES)
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        for k in ('dataset_manifests','metric_policy','class_names'):
            print('note: config differs (allowed for comparison):',k) if state['config'].get(k)!=cfg[k] else None
        eval_epoch=state['epoch'];del state
    args.output.mkdir(parents=True,exist_ok=bool(args.resume))
    model,classes=build(cfg,'current',checkpoint)
    if args.command=='evaluate':
        result=evaluation.evaluate(model,cfg,args.output,eval_epoch,data,classes,args.split,prior,core)
        core.write_json(args.output/'metrics.json',result)
        print(json.dumps(dict(event='final_test_complete',epoch=eval_epoch,macro_f1=result['macro_f1'],mask_mAP=result['mask_mAP'])),flush=True);return
    # User explicitly requested the preserved prior validation best, all six class weights intact.
    if not args.resume:
        source_state=torch.load(source,map_location='cpu',weights_only=False)
        head=[v for k,v in source_state['state_dict'].items() if 'bbox_head.rtm_cls.' in k]
        if not head or any(v.shape[0]!=6 for v in head):raise ValueError('Initialization is not a six-class checkpoint')
        if source_state['epoch']!=init['source_epoch'] or list(source_state['meta']['dataset_meta']['classes'])!=CLASSES:raise ValueError('Initial best provenance mismatch')
        if any(not torch.equal(value.detach().cpu(),source_state['state_dict'][key].detach().cpu()) for key,value in model.state_dict().items()):raise ValueError('Model weights differ from preserved best')
        del source_state,head
    opt=optimizer_for(model,cfg);scaler=torch.amp.GradScaler('cuda',init_scale=128,enabled=True)
    start=0;best=-1.;stale=0
    if args.resume:
        state=torch.load(args.resume,map_location='cpu',weights_only=False);restore(state,model,opt,scaler,cfg)
        start=state['epoch'];best=state['best'];stale=state['stale'];del state
    else:
        baseline=evaluation.evaluate(model,cfg,args.output,0,valid,classes,'valid',prior,core)
        best=evaluation.selection_score(baseline)
        core.write_json(args.output/'initial_validation.json',baseline)
        save(args.output/'last.pth',model,opt,scaler,0,best,0,cfg,'current')
        save(args.output/'best.pth',model,opt,scaler,0,best,0,cfg,'current')
    core.write_json(args.output/'effective_config.json',dict(**cfg,dataset=str(args.dataset),epochs=args.epochs,
                  test_used_for_selection=False,selection='valid six-class equal-weight macro F1',
                  items_per_epoch=len(epoch_images(train[0]['images'],0))))
    pipe=prior.pipeline(cfg['side'],True)
    for epoch in range(start,args.epochs):
        images=epoch_images(train[0]['images'],epoch);core.train_mode(model)
        batches=[images[n:n+cfg['batch_size']] for n in range(0,len(images),cfg['batch_size'])]
        total=0.;overflows=0;seen_gt=0;began=time.monotonic()
        input_load_seconds=0.;input_wait_seconds=0.
        with prefetched_batches(batches,train[1],pipe,prior.load_batch) as input_batches:
            for n,items,data,load_seconds,wait_seconds in input_batches:
                prior.resource_guard(cfg,args.output)
                if n%cfg['accumulation']==0:opt.zero_grad(set_to_none=True)
                input_load_seconds+=load_seconds;input_wait_seconds+=wait_seconds
                for im,sample in zip(items,data['data_samples']):
                    if len(sample.gt_instances) and not (sample.gt_instances.masks.areas>0).all():raise RuntimeError('Mask disappeared: '+im['source_key'])
                data=model.data_preprocessor(data,training=True)
                model.bbox_head.assignment_audit=[];model.bbox_head.assignment_repairs=[]
                expand_training_boxes(data,cfg['min_training_box_side'])
                losses,loss,retried=training_loss(model,data,cfg['amp_dtype'])
                audits=model.bbox_head.assignment_audit[-len(items):]
                if sum(a['missing_count'] for a in audits):raise RuntimeError('Unassigned GT: '+str(audits))
                seen_gt+=sum(a['gt'] for a in audits)
                if retried:print(json.dumps(dict(event='fp32_loss_retry',epoch=epoch+1,batch=n+1)),flush=True)
                group_start=(n//cfg['accumulation'])*cfg['accumulation'];divisor=min(cfg['accumulation'],len(batches)-group_start)
                scaler.scale(loss/divisor).backward();total+=float(loss.detach())
                if (n+1)%cfg['accumulation']==0 or n+1==len(batches):
                    scaler.unscale_(opt);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
                    progress=(epoch+(n+1)/len(batches))/args.epochs
                    warmup=min(1.,(epoch*len(batches)+n+1)/cfg['warmup_batches'])
                    for group in opt.param_groups:group['lr']=cfg['lr']*(.1+.9*(1+math.cos(math.pi*progress))/2)*warmup
                    scale=scaler.get_scale();scaler.step(opt);scaler.update()
                    overflows=overflows+1 if scaler.get_scale()<scale else 0
                    if overflows>=6:raise RuntimeError('Six consecutive gradient overflows')
                if n%10==0:print(json.dumps(dict(epoch=epoch+1,batch=n+1,batches=len(batches),loss=float(loss.detach()),lr=opt.param_groups[0]['lr'],seconds=round(time.monotonic()-began,1),input_wait_seconds=round(input_wait_seconds,3),input_load_seconds=round(input_load_seconds,3))),flush=True)
                del loss,losses,data
        result=evaluation.evaluate(model,cfg,args.output,epoch+1,valid,classes,'valid',prior,core)
        score=evaluation.selection_score(result);improved=score>best
        stale=0 if improved else stale+1
        if improved:best=score
        save(args.output/'last.pth',model,opt,scaler,epoch+1,best,stale,cfg,'current')
        archive_checkpoint(args.output/'last.pth',epoch+1,interval=10)
        if improved:save(args.output/'best.pth',model,opt,scaler,epoch+1,best,stale,cfg,'current')
        record=dict(epoch=epoch+1,mean_loss=total/len(batches),valid=result,best_selection_score=best,
                    training_assignment=dict(unassigned_gt=0,seen_gt=seen_gt),input_pipeline=dict(mode='ordered_one_batch_cpu_prefetch',load_seconds=input_load_seconds,wait_seconds=input_wait_seconds),seconds=round(time.monotonic()-began,2))
        with (args.output/'epochs.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(dict(event='epoch_complete',epoch=epoch+1,mean_loss=record['mean_loss'],macro_f1=score,mask_mAP=result['mask_mAP'],best_selection_score=best,seconds=record['seconds'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['train','evaluate'])
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',type=Path);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--split',choices=['valid','test'],default='test');p.add_argument('--epochs',type=int,default=100)
    run(p.parse_args())


