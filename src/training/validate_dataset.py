"""Reload saved train/holdouts; independently compare derived masks with parent pixels."""
from pathlib import Path
import collections,hashlib,json,time
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from pycocotools import mask as mu
from all6_sampling import COUNTS,epoch_images
import all6_metrics as metrics

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def decode(r):
    if isinstance(r['counts'],list):r=mu.frPyObjects(r,*r['size'])
    return mu.decode(r)

def main():
    cv2.setNumThreads(1);here=Path(__file__).resolve().parent;p=json.loads((here/'paths.json').read_text())
    dataset=Path(p['dataset']);source=Path(p['source_dataset']);start=time.monotonic()
    for root in (dataset,source):
        for name,digest in json.loads((root/'manifest_hashes.json').read_text()).items():assert sha(root/name)==digest,name
    train=json.loads((dataset/'current_train.coco.json').read_text());base=json.loads((source/'train_originals.coco.json').read_text())
    aa=collections.defaultdict(list);ba=collections.defaultdict(list);grouped=collections.defaultdict(list)
    for a in train['annotations']:aa[a['image_id']].append(a)
    for a in base['annotations']:ba[a['image_id']].append(a)
    parents={i['source_key']:i for i in base['images']}
    for im in train['images']:grouped[im['source_key']].append(im)
    assert set(grouped)==set(parents) and len(train['images'])==4420
    held=[]
    for split in ('valid','test'):
        assert sha(source/f'imx219_{split}.coco.json')==sha(dataset/f'imx219_{split}.coco.json')
        held.extend(metrics.load_evaluation(dataset,split,metrics.CLASSES)[0]['images'])
    for field in ('source_key','allocation_unit','image_sha256'):
        assert not {i[field] for i in train['images']}&{i[field] for i in held},field
    def verify_parent(item):
        key,ims=item
        nmask=0;class_totals=collections.Counter();cached_geometry=0
        parent=parents[key];original=ba[parent['id']];native={a['source_object_id']:decode(a['segmentation']) for a in original}
        assert len(native)==len(original)
        expected_cache={}
        assert len(ims)==10 and sum(not i['augmented'] for i in ims)==1
        assert collections.Counter(i['sample_view'] for i in ims)=={'full':6,'crop':4}
        for im in ims:
            assert im['eligible'] and im['split']=='train' and im['dataset_domain']=='imx219'
            assert im.get('supervision_boxes_xyxy') is None and not im.get('uncertain_boxes_xyxy')
            assert sha(im['file_name'])==im['image_sha256']
            for field in ('physical_capture_episode','capture_session','range_group'):assert im[field]==parent[field]
            if not im['augmented']:
                assert im['image_sha256']==parent['image_sha256']
                clean=lambda a:{k:v for k,v in a.items() if k not in ('id','image_id')}
                assert sorted(map(clean,aa[im['id']]),key=lambda a:a['source_object_id'])==sorted(map(clean,original),key=lambda a:a['source_object_id'])
                class_totals.update(a['category_id'] for a in original)
                continue
            assert im['parent_sha256']==parent['image_sha256'] and im['parent_id'] in {j['id'] for j in ims if not j['augmented']}
            x,y,w,h=im['crop_window_xywh'];flip=im['augmentation_parameters']['horizontal_flip']
            assert 0<=x and 0<=y and x+w<=parent['width'] and y+h<=parent['height']
            assert (im['width'],im['height'])==(w,h)
            geometry=(x,y,w,h,flip)
            if geometry not in expected_cache:
                expected={}
                for a in original:
                    m=native[a['source_object_id']][y:y+h,x:x+w]
                    if not m.any():continue
                    if a['category_id']!=6:assert int(m.sum())==a['area'],('target cut',key)
                    if flip:m=m[:,::-1]
                    scale=1600/max(w,h);small=cv2.resize(m,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_NEAREST)
                    assert small.any(),('disappeared at model size',key,a['source_object_id'])
                    yy,xx=np.nonzero(m);r=mu.encode(np.asfortranarray(m));r['counts']=r['counts'].decode()
                    expected[a['source_object_id']]=(a['category_id'],int(m.sum()),[int(xx.min()),int(yy.min()),int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1)],r)
                expected_cache[geometry]=expected;cached_geometry+=1
            expected=expected_cache[geometry];actual=aa[im['id']]
            assert len(actual)==len(expected)
            for a in actual:
                assert (a['category_id'],a['area'],a['bbox'],a['segmentation'])==expected[a['source_object_id']],(key,a['source_object_id'])
                nmask+=1
        return nmask,cached_geometry,class_totals
    nmask=0;class_totals=collections.Counter();cached_geometry=0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for n,(nm,ng,cc) in enumerate(pool.map(verify_parent,sorted(grouped.items())),1):
            nmask+=nm;cached_geometry+=ng;class_totals.update(cc)
            if n%40==0:print(json.dumps(dict(event='verify_parent_pixels',done=n,total=len(grouped),seconds=round(time.monotonic()-start))),flush=True)
    assert set(class_totals)==set(range(1,7))
    exposure=[]
    for epoch in range(100):
        chosen=epoch_images(train['images'],epoch)
        assert len(chosen)==1900 and {i['source_key'] for i in chosen}==set(parents)
        if epoch==0:
            exposure=dict(buckets=dict(collections.Counter(i['sampling_bucket'] for i in chosen)),
                ranges=dict(collections.Counter(i['range_group'] for i in chosen)),
                classes=dict(collections.Counter(a['category_id'] for i in chosen for a in aa[i['id']])) )
    for split,name in [('train','current_train.coco.json'),('valid','imx219_valid.coco.json'),('test','imx219_test.coco.json')]:
        assert (dataset/(split+'set')/'annotations.coco.json').resolve()==dataset/name
    report=dict(result='PASS',train_originals=442,train_items=4420,valid=55,test=56,source_labels_preserved=sum(class_totals.values()),
      derived_masks_pixel_verified=nmask,unique_derived_geometries=cached_geometry,all_images_sha_verified=True,
      target_partial_cuts=0,mask_loss_at_1600=0,holdout_bytes_unchanged=True,split_image_allocation_unit_overlap=0,physical_scenes_shared=True,independent_evaluation=False,
      source_capture_session_overlap_declared=json.loads((source/'evaluation_policy.json').read_text())['source_capture_session_overlap'],
      all_100_epochs_cover_all_442_sources=True,epoch_items=1900,first_epoch_exposure=exposure,seconds=round(time.monotonic()-start,2))
    (here/'dataset_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
