"""Six-class F1/AP from a single heldout prediction pass; test never selects best."""
import collections, copy, gzip, hashlib, json
from pathlib import Path
import numpy as np
from pycocotools import mask as mu
from coco_map_tables import summarize as coco_summary
POLICY_VERSION='all6_ratio811_shared_scene_best_history_20261029_v1'
CLASSES=['water','bolt','nut','oil','paint','reflection']

def validate_rows(data,policy,split):
    if split not in ('valid','test') or policy['sensor']!='IMX219': raise ValueError('Invalid evaluation scope')
    if data['categories'] != [dict(id=i+1,name=c) for i,c in enumerate(CLASSES)]: raise ValueError('Class mapping changed')
    shared=policy.get('allow_shared_physical_scenes',False)
    if shared and (policy.get('version')!='imx219_per_class_811_shared_scene_20261029_v1' or policy.get('independent_evaluation') is not False): raise ValueError('Unacknowledged shared-scene policy')
    expected=policy['splits'][split]['expected_frames']; keys=[i['source_key'] for i in data['images']]
    if len(keys)!=len(set(keys)) or set(keys)!=set(expected): raise ValueError('Heldout membership changed')
    for im in data['images']:
        key=im['source_key']; ref=expected[key]
        if (im.get('dataset_domain')!='imx219' or im.get('split')!=split or im.get('augmented') or im.get('parent_sha256')
            or not im.get('eligible') or not im.get('full_frame_review_confirmed')
            or im.get('supervision_boxes_xyxy') is not None or im.get('uncertain_boxes_xyxy')
            or key in policy['training_source_keys']
            or (not shared and (im['group'] in policy['training_groups'] or im['physical_capture_episode'] in policy['training_groups']))
            or (shared and im.get('allocation_unit') in policy['training_allocation_units'])
            or im['image_sha256'] in policy['training_hashes']
            or any(im.get(k)!=v for k,v in ref.items())):
            raise ValueError('Forbidden evaluation frame: '+key)
    counts=collections.Counter(a['category_id'] for a in data['annotations'])
    if set(counts)!=set(range(1,7)): raise ValueError('Positive GT missing from six-class evaluation')
    ids={i['id'] for i in data['images']}
    if any(a['image_id'] not in ids for a in data['annotations']): raise ValueError('Orphan annotation')

def load_evaluation(dataset,split,classes):
    if list(classes)!=CLASSES: raise ValueError('Class order mismatch')
    dataset=Path(dataset); hashes=json.loads((dataset/'manifest_hashes.json').read_text())
    def checked(name):
        path=dataset/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=hashes[name]: raise ValueError('Changed manifest: '+name)
        return json.loads(path.read_text())
    policy=checked('evaluation_policy.json'); data=checked(f'imx219_{split}.coco.json')
    validate_rows(data,policy,split)
    for im in data['images']:
        if hashlib.sha256(Path(im['file_name']).read_bytes()).hexdigest()!=im['image_sha256']:raise ValueError('Changed evaluation image')
    data['evaluation_policy']=policy; aa=collections.defaultdict(list)
    for a in data['annotations']:aa[a['image_id']].append(a)
    return data,aa

def selection_score(result):
    values=[]
    for c in CLASSES:
        m=result['classes'][c]
        if m['gt_instances']<=0 or m['f1'] is None:raise ValueError('Missing positive GT: '+c)
        values.append(m['f1'])
    return sum(values)/6
def class_counts(ground_truth,predictions,scores,threshold=.5):
    """Instance F1 matches in score order; pixel IoU uses class-wise mask unions."""
    # Reviewed snapshots contain both COCO list-count and compressed RLE.
    # Normalize in memory only; retain original annotation files and pixels.
    normalize=lambda r:mu.frPyObjects(r,*r['size']) if isinstance(r['counts'],list) else r
    ground_truth=[normalize(r) for r in ground_truth]
    predictions=[normalize(r) for r in predictions]
    matches=set();tp=0;matched_iou=0.
    if ground_truth and predictions:
        ious=mu.iou(predictions,ground_truth,[0]*len(ground_truth))
        for p in sorted(range(len(predictions)),key=lambda n:-scores[n]):
            candidates=[(float(ious[p,g]),g) for g in range(len(ground_truth)) if g not in matches]
            if candidates:
                value,g=max(candidates)
                if value>=threshold:matches.add(g);tp+=1;matched_iou+=value
    truth=mu.merge(ground_truth) if ground_truth else None
    pred=mu.merge(predictions) if predictions else None
    if truth is not None and pred is not None:
        intersection=int(mu.area(mu.merge([truth,pred],intersect=True)))
        union=int(mu.area(mu.merge([truth,pred])))
    else:
        intersection=0;union=int(mu.area(truth if truth is not None else pred)) if truth is not None or pred is not None else 0
    return dict(tp=tp,fp=len(predictions)-tp,fn=len(ground_truth)-tp,
                intersection_pixels=intersection,union_pixels=union,matched_iou_sum=matched_iou)


def summarize(counts):
    tp,fp,fn=(counts[k] for k in ('tp','fp','fn'))
    return dict(counts,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
                f1=2*tp/(2*tp+fp+fn) if tp+fn else None,gt_instances=tp+fn,positive_class_evaluation=bool(tp+fn),
                mask_iou=counts['intersection_pixels']/counts['union_pixels'] if counts['union_pixels'] else None,
                matched_mean_mask_iou=counts['matched_iou_sum']/tp if tp else None)



def summarize_distances(per_image, classes):
    fields=('tp','fp','fn','intersection_pixels','union_pixels','matched_iou_sum')
    buckets=collections.defaultdict(list)
    for row in per_image:
        buckets[row.get('range_group','unspecified')].append(row)
    output={}
    for distance, rows in buckets.items():
        metrics={}
        for name in classes:
            counts=collections.Counter()
            for row in rows:
                counts.update({k:row['classes'][name][k] for k in fields})
            metrics[name]=summarize(counts)
        output[distance]=dict(images=len(rows),classes=metrics)
    return output

def evaluate(model,cfg,out,epoch,dataset,classes,split,prior,core):
    import torch
    d,aa=dataset;validate_rows(d,d['evaluation_policy'],split)
    pipe=prior.pipeline(cfg['side'],False);model.eval()
    sums={c:collections.Counter() for c in classes};per_image=[];predictions=[]
    # AP needs the ranking below the F1 operating threshold; one inference feeds both.
    saved=[]
    for tc in {id(c):c for c in (model.test_cfg,model.bbox_head.test_cfg)}.values():
        saved.append((tc,copy.deepcopy(dict(tc))));tc.score_thr=.001;tc.max_per_img=100;tc.nms_pre=1000
    try:
        for n,im in enumerate(d['images']):
            prior.resource_guard(cfg,out)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                pred=model.test_step(prior.load_batch([im],aa,pipe))[0].pred_instances.cpu()
            if tuple(pred.masks.shape[-2:])!=(im['height'],im['width']):raise ValueError('Prediction mask is not at native resolution')
            if not torch.isfinite(pred.scores).all():raise ValueError('Nonfinite predictions')
            encoded=[]
            for j,m in enumerate(pred.masks.numpy()):
                r=mu.encode(np.asfortranarray(m.astype(np.uint8)));r['counts']=r['counts'].decode();encoded.append(r)
                predictions.append(dict(image_id=im['id'],category_id=int(pred.labels[j])+1,score=float(pred.scores[j]),segmentation=r))
            row=dict(source_key=im['source_key'],range_group=im.get('range_group','unknown'),classes={})
            for cid,c in enumerate(classes,1):
                gt=[a['segmentation'] for a in aa[im['id']] if a['category_id']==cid]
                ix=[j for j in range(len(pred)) if int(pred.labels[j])==cid-1 and float(pred.scores[j])>=cfg['score_threshold']]
                counts=class_counts(gt,[encoded[j] for j in ix],[float(pred.scores[j]) for j in ix],cfg['mask_iou_threshold'])
                sums[c].update(counts);row['classes'][c]=summarize(counts)
            per_image.append(row)
            if n%20==0:print(json.dumps(dict(event='evaluation',epoch=epoch,split=split,image=n+1,total=len(d['images']))),flush=True)
            del pred,encoded
        metrics={c:summarize(v) for c,v in sums.items()}
        gt={k:v for k,v in d.items() if k!='evaluation_policy'}
        maps=coco_summary(gt,predictions,'segm')
        if set(maps['all']['supported_classes'])!=set(CLASSES):raise ValueError('COCO omitted a class')
        negative=[r for r in per_image if sum(r['classes'][c]['gt_instances'] for c in CLASSES[:5])==0]
        fp=sum(r['classes'][c]['fp'] for r in negative for c in CLASSES[:5])
        result=dict(epoch=epoch,split=split,scope='imx219_'+split,images=len(d['images']),classes=metrics,
          macro_f1=sum(metrics[c]['f1'] for c in classes)/6,
          macro_mask_iou=sum(metrics[c]['mask_iou'] for c in classes)/6,
          mask_mAP=maps['all']['mAP'],mask_AP50=maps['all']['AP50'],coco_mask_ap=maps,
          per_distance=summarize_distances(per_image,classes),per_image=per_image,
          negative_images=len(negative),hazard_false_positives_on_negative_images=fp,
          hazard_fp_per_negative_image=fp/len(negative) if negative else None,
          negative_images_with_hazard_fp=sum(any(r['classes'][c]['fp'] for c in CLASSES[:5]) for r in negative),
          score_threshold=cfg['score_threshold'],mask_iou_match_threshold=cfg['mask_iou_threshold'],ap_score_floor=.001,
          metric_policy=POLICY_VERSION,test_used_for_selection=False,evaluation_limit=d['evaluation_policy']['limitations'],
          independent_evaluation=False,physical_scenes_shared=True,prior_model_exposure=d['evaluation_policy']['prior_model_exposure'][split])
        assert abs(result['macro_f1']-selection_score(result))<1e-12
        folder=Path(out)/'validation_metrics';folder.mkdir(exist_ok=True)
        core.write_json(folder/f'{split}_epoch_{epoch:03d}.json',result)
        temp=folder/'latest_predictions.json.gz.tmp'
        with gzip.open(temp,'wt') as f:json.dump(predictions,f)
        temp.replace(folder/'latest_predictions.json.gz')
        return result
    finally:
        for tc,old in saved:tc.clear();tc.update(old)
        core.train_mode(model);core.release_cache()


