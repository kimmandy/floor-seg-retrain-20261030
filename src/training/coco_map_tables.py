"""COCO AP summaries by original-image distance and semantic class."""
import contextlib,copy,io
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def optional_mean(a):
 a=np.asarray(a);a=a[a>=0]
 return float(a.mean()) if a.size else None

def summarize(ground_truth,predictions,iou_type='segm'):
 """AP averages 10 IoUs (.50:.05:.95), 101 recalls, maxDets=100."""
 with contextlib.redirect_stdout(io.StringIO()):
  gt=COCO();gt.dataset=copy.deepcopy(ground_truth);gt.dataset.setdefault('info',{});gt.createIndex()
  if predictions:dt=gt.loadRes(copy.deepcopy(predictions))
  else:
   dt=COCO();dt.dataset=dict(info={},images=copy.deepcopy(gt.dataset['images']),categories=copy.deepcopy(gt.dataset['categories']),annotations=[]);dt.createIndex()
  image_ids=[i['id'] for i in ground_truth['images']]
  groups={'all':image_ids}
  for distance in sorted({i.get('range_group','unknown') for i in ground_truth['images']}):
   groups[distance]=[i['id'] for i in ground_truth['images'] if i.get('range_group')==distance]
  result={}
  for distance,ids in groups.items():
   classes={};positives={c['id']:sum(a['category_id']==c['id'] and a['image_id'] in set(ids) for a in ground_truth['annotations']) for c in ground_truth['categories']}
   if ids:
    evaluator=COCOeval(gt,dt,iou_type);evaluator.params.imgIds=ids;evaluator.params.catIds=[c['id'] for c in ground_truth['categories']];evaluator.params.maxDets=[1,10,100]
    evaluator.evaluate();evaluator.accumulate();evaluator.summarize()
    precision=evaluator.eval['precision'];i50=int(np.flatnonzero(np.isclose(evaluator.params.iouThrs,.5))[0]);i75=int(np.flatnonzero(np.isclose(evaluator.params.iouThrs,.75))[0])
    for k,c in enumerate(ground_truth['categories']):
     p=precision[:,:,k,0,-1]
     classes[c['name']]=dict(gt_instances=positives[c['id']],AP=optional_mean(p),AP50=optional_mean(p[i50]),AP75=optional_mean(p[i75]))
    overall=dict(mAP=optional_mean(precision[:,:,:,0,-1]),AP50=optional_mean(precision[i50,:,:,0,-1]),AP75=optional_mean(precision[i75,:,:,0,-1]),
                 AP_small=optional_mean(precision[:,:,:,1,-1]),AP_medium=optional_mean(precision[:,:,:,2,-1]),AP_large=optional_mean(precision[:,:,:,3,-1]))
   else:
    classes={c['name']:dict(gt_instances=0,AP=None,AP50=None,AP75=None) for c in ground_truth['categories']};overall=dict(mAP=None,AP50=None,AP75=None,AP_small=None,AP_medium=None,AP_large=None)
   result[distance]=dict(images=len(ids),gt_instances=sum(positives.values()),**overall,classes=classes,
                         supported_classes=[name for name,v in classes.items() if v['gt_instances']>0])
 return result
