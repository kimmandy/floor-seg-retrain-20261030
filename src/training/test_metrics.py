"""Known-answer metric tests and evaluation-contamination rejection."""
import collections,copy,json
from pathlib import Path
import numpy as np
from pycocotools import mask as mu
import all6_metrics as metrics
from coco_map_tables import summarize as ap

def main():
    m=np.zeros((20,20),np.uint8);m[5:12,6:13]=1
    r=mu.encode(np.asfortranarray(m));r['counts']=r['counts'].decode()
    assert metrics.summarize(metrics.class_counts([r],[r],[.9]))['f1']==1.
    q=metrics.summarize(metrics.class_counts([r],[r,r],[.9,.8]))
    assert (q['tp'],q['fp'],q['fn'])==(1,1,0) and abs(q['f1']-2/3)<1e-12
    assert metrics.summarize(metrics.class_counts([r],[],[]))['f1']==0.
    d=dict(images=[dict(id=1,width=20,height=20,range_group='far')],categories=[dict(id=i+1,name=c) for i,c in enumerate(metrics.CLASSES)],
           annotations=[dict(id=i,image_id=1,category_id=i,segmentation=r,area=49,bbox=[6,5,7,7],iscrowd=0) for i in range(1,7)])
    p=[dict(image_id=1,category_id=i,score=.9,segmentation=r) for i in range(1,7)]
    perfect=ap(d,p)['all'];zero=ap(d,[])['all']
    assert abs(perfect['mAP']-1)<1e-12 and zero['mAP']==0
    result={'classes':{c:dict(gt_instances=1,f1=(0 if c=='water' else 1)) for c in metrics.CLASSES}}
    assert abs(metrics.selection_score(result)-5/6)<1e-12
    result['classes']['water']['gt_instances']=0
    try:metrics.selection_score(result)
    except ValueError:pass
    else:raise AssertionError('Missing class silently accepted')
    root=Path(__file__).resolve().parent;path=json.loads((root/'paths.json').read_text())
    base=Path(path['source_dataset']);policy=json.loads((base/'evaluation_policy.json').read_text())
    data=json.loads((base/'imx219_valid.coco.json').read_text());metrics.validate_rows(data,policy,'valid')
    # Regression: new holdouts include uncompressed RLE from completed labels.
    format_counts=collections.Counter()
    for split in ('valid','test'):
        actual=json.loads((base/f'imx219_{split}.coco.json').read_text())
        for a in actual['annotations']:
            raw=a['segmentation']
            compressed=mu.frPyObjects(raw,*raw['size']) if isinstance(raw['counts'],list) else raw
            counts=metrics.class_counts([raw],[compressed],[.9])
            assert (counts['tp'],counts['fp'],counts['fn'])==(1,0,0)
            assert counts['intersection_pixels']==counts['union_pixels']==a['area']
            format_counts['list' if isinstance(raw['counts'],list) else 'compressed']+=1
    assert format_counts['list']>0 and format_counts['compressed']>0
    # COCO AP also accepts native list-count GT without rewriting the source.
    flat=m.flatten(order='F');runs=[];previous=0;length=0
    for value in flat:
        if value==previous:length+=1
        else:runs.append(length);length=1;previous=int(value)
    runs.append(length)
    list_rle=dict(size=list(m.shape),counts=runs)
    dc=copy.deepcopy(d)
    for a in dc['annotations']:a['segmentation']=list_rle
    assert abs(ap(dc,p)['all']['mAP']-1)<1e-12
    rejected=[]
    for key,val in [('dataset_domain','usb'),('augmented',True),('split','train'),('allocation_unit',policy['training_allocation_units'][0])]:
        wrong=copy.deepcopy(data);wrong['images'][0][key]=val
        try:metrics.validate_rows(wrong,policy,'valid')
        except ValueError:rejected.append(key)
        else:raise AssertionError(key)
    wrong=copy.deepcopy(data);wrong['annotations']=[a for a in wrong['annotations'] if a['category_id']!=5]
    try:metrics.validate_rows(wrong,policy,'valid')
    except ValueError:rejected.append('missing_paint')
    else:raise AssertionError('missing paint')
    badpolicy=copy.deepcopy(policy);badpolicy['independent_evaluation']=True
    try:metrics.validate_rows(data,badpolicy,'valid')
    except ValueError:rejected.append('false_independence_claim')
    else:raise AssertionError('Shared scenes claimed independent')
    output=dict(result='PASS',perfect_mask_ap=perfect['mAP'],empty_predictions_ap=zero['mAP'],duplicate_prediction_penalized=True,
                six_class_equal_weight_test=True,rejected=rejected,rle_format_pixel_and_metric_equivalence=dict(format_counts))
    (root/'metric_tests.json').write_text(json.dumps(output,indent=2));print(json.dumps(output))
if __name__=='__main__':main()
