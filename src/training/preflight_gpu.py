"""Real CUDA loss/backward/update, checkpoint state roundtrip, full valid inference."""
import argparse,collections,copy,fcntl,hashlib,json,random,time
from pathlib import Path
import numpy as np
import torch
import train_all6 as train
import all6_metrics as metrics
from all6_sampling import bucket,epoch_images

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--originals',action='store_true');args=parser.parse_args()
    here=Path(__file__).resolve().parent;p=json.loads((here/'paths.json').read_text())
    lock=open('/data/baseline/training/runs/.training.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    dataset=Path(p['source_dataset'] if args.originals else p['dataset']);tag='originals' if args.originals else 'augmented'
    out=here/('preflight_'+tag);out.mkdir(exist_ok=False)
    torch.set_num_threads(4);random.seed(20261028);np.random.seed(20261028);torch.manual_seed(20261028)
    cfg=train.configuration('current');cfg['early_stopping_patience']=0;cfg['schedule_epochs']=100;cfg['metric_policy']=metrics.POLICY_VERSION
    cfg['dataset_manifests']=json.loads((dataset/'manifest_hashes.json').read_text())
    source=Path(p['checkpoint']);ck=torch.load(source,map_location='cpu',weights_only=False)
    classifiers=[v for k,v in ck['state_dict'].items() if 'bbox_head.rtm_cls.' in k]
    assert classifiers and all(v.shape[0]==6 for v in classifiers);del ck,classifiers
    model,classes=train.build(cfg,'current',source)
    if args.originals:
        data=train.prior.read_data(dataset/'train_originals.coco.json')
        for im in data[0]['images']:im['sampling_bucket']=bucket(im,data[1][im['id']])
    else:data=train.load_data(dataset,'current','train')
    valid=metrics.load_evaluation(dataset,'valid',classes);test=metrics.load_evaluation(dataset,'test',classes)
    candidates=data[0]['images'];chosen=[]
    for b in sorted({i['sampling_bucket'] for i in candidates}):
        for view in sorted({i['sample_view'] for i in candidates if i['sampling_bucket']==b}):
            pool=[i for i in candidates if i['sampling_bucket']==b and i['sample_view']==view]
            chosen.append(min(pool,key=lambda im:min((a['area'] for a in data[1][im['id']]),default=0)))
    chosen.append(max(candidates,key=lambda im:len(data[1][im['id']])))
    unique={i['id']:i for i in chosen};chosen=list(unique.values())
    pipe=train.prior.pipeline(1600,True);opt=train.optimizer_for(model,cfg);scaler=torch.amp.GradScaler('cuda',init_scale=128,enabled=True);audit=[]
    for n in range(0,len(chosen),2):
        ims=chosen[n:n+2];opt.zero_grad(set_to_none=True);batch=train.prior.load_batch(ims,data[1],pipe)
        for im,s in zip(ims,batch['data_samples']):
            assert (s.gt_instances.masks.areas>0).all(),im['source_key']
            assert s.metainfo.get('supervision_boxes_input') is None and not s.metainfo.get('uncertain_boxes_input')
        batch=model.data_preprocessor(batch,training=True);model.bbox_head.assignment_audit=[];model.bbox_head.assignment_repairs=[]
        train.expand_training_boxes(batch,20);losses,loss,retried=train.training_loss(model,batch,'bfloat16')
        scaler.scale(loss).backward();scaler.unscale_(opt)
        gradients=[x.grad for x in model.parameters() if x.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert not any(a['missing_count'] for a in model.bbox_head.assignment_audit[-len(ims):])
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.);assert torch.isfinite(norm)
        before=next(model.parameters()).detach().clone();scaler.step(opt);scaler.update()
        assert not torch.equal(before,next(model.parameters()).detach()),'optimizer did not update'
        row=dict(images=[i['source_key'] for i in ims],views=[i['sample_view'] for i in ims],loss=float(loss.detach()),finite_gradients=len(gradients),unassigned_gt=0)
        audit.append(row);print(json.dumps(dict(event='gpu_backward',**row)),flush=True);del loss,losses,batch,before
    train.save(out/'roundtrip.pth',model,opt,scaler,3,.1,0,cfg,'current')
    state=torch.load(out/'roundtrip.pth',map_location='cpu',weights_only=False)
    next_random=(random.random(),float(np.random.rand()),torch.rand(2),torch.rand(2,device='cuda'))
    with torch.no_grad():next(model.parameters()).add_(.1)
    train.restore(state,model,opt,scaler,cfg)
    assert torch.equal(next(model.parameters()).cpu(),state['state_dict'][next(iter(model.state_dict()))])
    restored_random=(random.random(),float(np.random.rand()),torch.rand(2),torch.rand(2,device='cuda'))
    assert next_random[:2]==restored_random[:2] and all(torch.equal(a,b) for a,b in zip(next_random[2:],restored_random[2:]))
    assert opt.state_dict()['param_groups']==state['optimizer']['param_groups'] and scaler.state_dict()==state['scaler']
    for paramid,values in opt.state_dict()['state'].items():
        for key,value in values.items():
            original=state['optimizer']['state'][paramid][key]
            assert torch.equal(value.cpu(),original.cpu()) if isinstance(value,torch.Tensor) else value==original
    bad=copy.deepcopy(cfg);bad['lr']*=2
    try:train.restore(state,model,opt,scaler,bad)
    except ValueError:pass
    else:raise AssertionError('Changed-LR resume accepted')
    bad=copy.deepcopy(cfg);bad['dataset_manifests']={'tampered':True}
    try:train.restore(state,model,opt,scaler,bad)
    except ValueError:pass
    else:raise AssertionError('Changed-dataset resume accepted')
    del state
    # Direct BF16 NMS with nonempty detections protects against the prior FP16 CUDA NMS failure.
    from mmcv.ops import batched_nms
    boxes=torch.tensor([[0,0,20,20],[1,1,21,21]],device='cuda',dtype=torch.float32)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        kept=batched_nms(boxes,torch.tensor([.9,.8],device='cuda'),torch.zeros(2,device='cuda',dtype=torch.long),dict(type='nms',iou_threshold=.5))[1]
    assert kept.tolist()==[0]
    if not args.originals:
        # Exercise installed AMP NMS through real model inference with nonempty output.
        configs=[]
        for tc in {id(c):c for c in (model.test_cfg,model.bbox_head.test_cfg)}.values():
            configs.append((tc,copy.deepcopy(dict(tc))));tc.score_thr=0.;tc.max_per_img=8;tc.nms_pre=50
        model.eval()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            forced=model.test_step(train.prior.load_batch([valid[0]['images'][0]],valid[1],train.prior.pipeline(1600,False)))[0].pred_instances
        assert len(forced)>0 and torch.isfinite(forced.scores).all()
        forced_count=len(forced);del forced
        for tc,old in configs:
            tc.clear();tc.update(old);assert dict(tc)==old
        train.core.train_mode(model)
    before_config=copy.deepcopy(dict(model.test_cfg))
    result=metrics.evaluate(model,cfg,out,0,valid,classes,'valid',train.prior,train.core)
    assert dict(model.test_cfg)==before_config,'Evaluation did not restore inference configuration'
    validation=dict(images=result['images'],six_classes=list(result['classes']),macro_f1=result['macro_f1'],mask_mAP=result['mask_mAP'],negative_images=result['negative_images'])
    if not args.originals:
        assert hashlib.sha256((dataset/'imx219_valid.coco.json').read_bytes()).hexdigest()==hashlib.sha256((Path(p['source_dataset'])/'imx219_valid.coco.json').read_bytes()).hexdigest()
        validation['identical_validation_bytes_to_originals']=True
        for epoch in range(100):assert len(epoch_images(data[0]['images'],epoch))==1900
    report=dict(result='PASS',scope=tag,checkpoint=str(source),checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_classifier_classes=6,loss_backward=audit,optimizer_scaler_rng_roundtrip=True,changed_lr_resume_rejected=True,changed_dataset_resume_rejected=True,
        nonempty_nms_pass=True,validation=validation,test_membership_only=len(test[0]['images']),trial_weights_used_for_training=False,
        peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    if not args.originals:report['real_model_fp16_nms_nonempty_predictions']=forced_count
    (here/'preflight_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
