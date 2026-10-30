"""Distance-balanced exposure and input-only boxes for tiny mask targets."""
import collections,random
POLICY='imx_distance_balanced_minbox20_coverage_v1'

def expand_training_boxes(data,minimum=20):
    """Keep source labels/masks intact; expand loss-target boxes in resized pixels."""
    import torch
    if minimum<=0:return
    for sample in data['data_samples']:
        gt=sample.gt_instances
        if not len(gt):continue
        wrapped=hasattr(gt.bboxes,'tensor');boxes=gt.bboxes.tensor if wrapped else gt.bboxes
        height,width=sample.metainfo['img_shape'][:2]
        center=(boxes[:,:2]+boxes[:,2:])/2
        size=(boxes[:,2:]-boxes[:,:2]).clamp(min=minimum)
        size[:,0].clamp_(max=width);size[:,1].clamp_(max=height)
        lower=(center-size/2).clamp(min=0)
        lower[:,0]=torch.minimum(lower[:,0],width-size[:,0]);lower[:,1]=torch.minimum(lower[:,1],height-size[:,1])
        result=torch.cat([lower,lower+size],dim=1)
        if wrapped:gt.bboxes=type(gt.bboxes)(result)
        else:gt.bboxes=result

def bucket(im):
    if im.get('dataset_domain')!='imx219':return 'usb'
    group=im['group'].removeprefix('imx219_20261021_')
    if group=='empty':return 'empty'
    target='spill' if group.startswith('spill') else 'fastener'
    return target+'_'+im['range_group']

def epoch_images(images,epoch):
    groups=collections.defaultdict(list)
    for im in images:groups[bucket(im)].append(im)
    expected={f'{c}_{d}' for c in ['fastener','spill'] for d in ['near','mid','far']}|{'empty','usb'}
    assert set(groups)==expected,set(groups)
    rng=random.Random(20261023+epoch);chosen=[]
    # Six object/range buckets each 120; background 60; USB originals 136.
    for key in sorted(groups):
        pool=sorted(groups[key],key=lambda i:i['id']);rng.shuffle(pool)
        count=len(pool) if key=='usb' else 60 if key=='empty' else 120
        # Rotate unselected halves between epochs; complete cycles before reuse.
        offset=(epoch*count)%len(pool)
        chosen.extend(pool[(offset+n)%len(pool)] for n in range(count))
    rng.shuffle(chosen)
    assert len(chosen)==916
    return chosen


def install_assignment_monitor(head):
    """Record final-forward GT coverage; never silently hide zero-positive targets."""
    import torch
    original=head._get_targets_single
    head.assignment_audit=[]
    def monitored(*args,**kwargs):
        result=original(*args,**kwargs)
        gt=args[4];sample=result[-1]
        counts=torch.bincount(sample.pos_assigned_gt_inds,minlength=len(gt))
        missing=(counts==0).nonzero().flatten()
        head.assignment_audit.append(dict(img_id=args[5].get('img_id'),gt=len(gt),missing_count=len(missing),missing_labels=gt.labels[missing].detach().cpu().tolist()))
        return result
    head._get_targets_single=monitored

def install_assignment_coverage(head):
    """Repair dynamic-k conflicts without taking another GT's sole positive."""
    import torch
    assigner=head.assigner
    original=assigner.dynamic_k_matching
    head.assignment_repairs=[]
    def repaired(cost,pairwise_ious,num_gt,valid_mask):
        initial_valid=valid_mask.clone()
        ious,indices=original(cost,pairwise_ious,num_gt,valid_mask)
        counts=torch.bincount(indices,minlength=num_gt)
        missing=(counts==0).nonzero().flatten()
        if not len(missing):return ious,indices
        n=cost.shape[0]
        if n<num_gt:
            head.assignment_repairs.append(dict(missing_before=len(missing),repaired=0,reason='fewer_candidate_priors_than_gt'))
            return ious,indices
        assignments=torch.full((n,),-1,dtype=torch.long,device=cost.device)
        assignments[valid_mask[initial_valid]]=indices
        repaired_iou=[]
        for gt in missing.tolist():
            donor=(assignments<0)|(counts[assignments.clamp_min(0)]>1)
            # Prefer intersecting boxes so quality-weighted classification/regression also receive signal.
            positive=donor&(pairwise_ious[:,gt]>0)
            if positive.any():donor=positive
            candidates=donor.nonzero().flatten()
            if not len(candidates):continue
            index=candidates[torch.argmin(cost[candidates,gt])]
            previous=int(assignments[index])
            if previous>=0:counts[previous]-=1
            assignments[index]=gt;counts[gt]+=1
            repaired_iou.append(float(pairwise_ious[index,gt]))
        foreground=assignments>=0
        valid_mask[initial_valid]=foreground
        indices=assignments[foreground]
        ious=pairwise_ious[foreground].gather(1,indices[:,None]).squeeze(1)
        head.assignment_repairs.append(dict(missing_before=len(missing),repaired=len(repaired_iou),assigned_iou=repaired_iou))
        return ious,indices
    assigner.dynamic_k_matching=repaired
