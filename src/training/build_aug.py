"""Train-only generation from the frozen 442/55/56 originals."""
import collections, concurrent.futures, copy, json, random, shutil, time
from pathlib import Path
import cv2
import aug_core as core
from all6_sampling import COUNTS, POLICY, bucket, epoch_images
HERE=Path(__file__).resolve().parent
P=json.loads((HERE/'paths.json').read_text())
BASE=Path(P['source_dataset'])
SEED=20261028
CACHE={}
CACHE_AA={}
def views(item):
 """Return [(image, annotations)] for one original; mirrors prepare_sixclass.augment_one."""
 cv2.setNumThreads(1);im,anns,out_dir=item;raw=cv2.imread(im['file_name']);assert raw is not None,im['file_name']
 assert raw.shape[:2]==(im['height'],im['width']) and core.sha(im['file_name'])==im['image_sha256'],im['source_key']
 masks=[core.decode(a['segmentation']) for a in anns];stats=collections.Counter()
 lost=[a['source_object_id'] for a,m in zip(anns,masks) if not core.fits_resize(m,im['width'],im['height'])]
 if lost:raise ValueError(('original target mask lost at 1600 resize',im['source_key'],lost))
 out=[(copy.deepcopy(im),copy.deepcopy(anns))];stats['original']+=1
 for j in range(9):
  crop=j>=5;flip=j in (3,4,6,8);rng=random.Random(SEED+im['id']*100+j)
  window,focus,fallback=core.choose_window(im,anns,j-5,rng) if crop else ((0,0,im['width'],im['height']),None,False)
  x,y,w,h=window
  # A clipped reflection fragment must not silently become unlabeled background.
  if any(m[y:y+h,x:x+w].any() and not core.fits_resize(m[y:y+h,x:x+w],w,h) for m in masks):
   window=(0,0,im['width'],im['height']); x,y,w,h=window; fallback=True
  rgb,params=core.render(raw[y:y+h,x:x+w],SEED+im['id']*100+j,j if not crop else 2,flip)
  labels=[]
  for a,m in zip(anns,masks):
   m=m[y:y+h,x:x+w];area=int(m.sum())
   if not area:continue
   if a['category_id'] in core.TARGETS and area!=a['area']:raise ValueError(('crop cut target',im['source_key'],a['source_object_id']))
   if flip:m=m[:,::-1]
   if not core.fits_resize(m,w,h):
    raise ValueError(('view target lost',im['source_key'],a['source_object_id'],j))
   labels.append(core.annotation(m,a['category_id'],a['source_object_id']))
  file=out_dir/f'{im["source_key"]}_{j:02d}.png'
  assert cv2.imwrite(str(file),rgb,[cv2.IMWRITE_PNG_COMPRESSION,1])
  ni=copy.deepcopy(im);ni.update(file_name=str(file),width=w,height=h,image_sha256=core.sha(file),augmented=True,
   parent_id=im['id'],parent_sha256=im['image_sha256'],augmentation=('crop_' if crop else 'optical_')+str(j-5 if crop else j),
   augmentation_parameters=params,sample_view='crop' if crop else 'full',crop_window_xywh=list(window),
   crop_focus_category=focus,full_frame_crop_fallback=fallback)
  for field in ('supervision_boxes_xyxy','uncertain_boxes_xyxy'):
   if field in im:ni[field]=core.transform_boxes(im[field],window,flip)
  out.append((ni,labels));stats['crop' if crop else 'optical']+=1
  if crop:stats['crop_'+str(w)]+=1;stats['full_frame_crop_fallback']+=fallback
 return out,dict(stats)

def main():
 global CACHE,CACHE_AA
 start=time.monotonic(); outdir=Path(P['dataset'])
 cache_root=Path('/data/baseline/datasets/imx219_all6_aug_20261028_1948')
 cache_manifest=json.loads((cache_root/'manifest_hashes.json').read_text())
 assert core.sha(cache_root/'current_train.coco.json')==cache_manifest['current_train.coco.json']
 cached=json.loads((cache_root/'current_train.coco.json').read_text())
 CACHE=collections.defaultdict(list);CACHE_AA=collections.defaultdict(list)
 for im in cached['images']:CACHE[im['source_key']].append(im)
 for a in cached['annotations']:CACHE_AA[a['image_id']].append(a)
 manifest=json.loads((BASE/'manifest_hashes.json').read_text())
 for n,h in manifest.items(): assert core.sha(BASE/n)==h, n
 base=json.loads((BASE/'train_originals.coco.json').read_text())
 by_image=collections.defaultdict(list)
 for a in base['annotations']: by_image[a['image_id']].append(a)
 images=copy.deepcopy(base['images'])
 policy=json.loads((BASE/'evaluation_policy.json').read_text())
 held=[im for s in ('valid','test') for im in json.loads((BASE/f'imx219_{s}.coco.json').read_text())['images']]
 for field in ('source_key','image_sha256','allocation_unit'):
  assert not {i[field] for i in images}&{i[field] for i in held},field
 for im in images:
  assert im['split']=='train' and not im['augmented'] and im['full_frame_review_confirmed']
  assert im.get('supervision_boxes_xyxy') is None and not im.get('uncertain_boxes_xyxy')
  im['source_sampling_bucket']=im['sampling_bucket']; im['sampling_bucket']=bucket(im,by_image[im['id']])
 present=collections.Counter(i['sampling_bucket'] for i in images)
 assert set(present)==set(COUNTS),present
 outdir.mkdir(parents=True,exist_ok=False); vdir=outdir/'views';vdir.mkdir()
 train={'images':[],'annotations':[],'categories':base['categories']};stats=collections.Counter()
 work=[(im,by_image[im['id']],vdir) for im in sorted(images,key=lambda i:i['id'])]
 with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
  for n,(views_out,counts) in enumerate(pool.map(views,work),1):
   stats.update(counts)
   for im,aa in views_out:
    im['id']=len(train['images'])+1;train['images'].append(im)
    for a in aa:train['annotations'].append(dict(a,id=len(train['annotations'])+1,image_id=im['id']))
   if n%20==0 or n==len(work): print(json.dumps(dict(event='augment',done=n,total=len(work),seconds=round(time.monotonic()-start))),flush=True)
 first={i['source_key']:i['id'] for i in train['images'] if not i['augmented']}
 for im in train['images']:
  if im['augmented']: im['parent_id']=first[im['source_key']]
 core.write(outdir/'current_train.coco.json',train)
 for s in ('valid','test'):
  shutil.copyfile(BASE/f'imx219_{s}.coco.json',outdir/f'imx219_{s}.coco.json')
  assert core.sha(BASE/f'imx219_{s}.coco.json')==core.sha(outdir/f'imx219_{s}.coco.json')
 policy['training_hashes']=sorted({im[k] for im in train['images'] for k in ('image_sha256','parent_sha256') if im.get(k)})
 policy['augmentation_plan']=POLICY
 core.write(outdir/'evaluation_policy.json',policy)
 plan=dict(policy=POLICY,originals=len(images),original_plus_optical_plus_crops=[1,5,4],
   crop_nominal_sides=[800,1024,1200,1024],crop_target_partial_cut='reject; enlarge or full-frame fallback',
   reflection_fragments='preserve; full-frame fallback if resized fragment would vanish',
   optical=dict(exposure=[.85,1.12],gamma=[.92,1.08],white_balance_rb=[.92,1.08],saturation=[.8,1.15],noise_sigma=[.4,1.5],lighting_gradient=[-.1,.1],horizontal_flip=True),
   epoch_quotas=COUNTS,items_per_epoch=sum(COUNTS.values()),full_crop_ratio=[1,1],
   every_source_exposed_each_epoch=True,unknown_distances_preserved=True,
   no_synthetic_pastes=True,no_test_or_valid_augmentation=True)
 core.write(outdir/'AUGMENTATION_PLAN.json',plan)
 sampling=[]
 for epoch in range(P['epochs']):
  chosen=epoch_images(train['images'],epoch)
  sampling.append(dict(epoch=epoch+1,items=len(chosen),unique_sources=len({i['source_key'] for i in chosen})))
 report=dict(result='PASS',source=str(BASE),originals=len(images),originals_by_bucket=dict(present),
   train_items=len(train['images']),train_annotations=len(train['annotations']),views=dict(stats),
   valid=55,test=56,epoch_items=sum(COUNTS.values()),target_or_reflection_masks_lost=0,
   holdout_bytes_unchanged=True,source_manifest=manifest,sampling_schedule=sampling,seconds=round(time.monotonic()-start))
 core.write(outdir/'build_report.json',report)
 names=['current_train.coco.json','imx219_valid.coco.json','imx219_test.coco.json','evaluation_policy.json','AUGMENTATION_PLAN.json','build_report.json']
 core.write(outdir/'manifest_hashes.json',{n:core.sha(outdir/n) for n in names})
 for split,name in [('train','current_train.coco.json'),('valid','imx219_valid.coco.json'),('test','imx219_test.coco.json')]:
  (outdir/(split+'set')).mkdir();(outdir/(split+'set')/'annotations.coco.json').symlink_to(outdir/name)
 print(json.dumps({k:v for k,v in report.items() if k not in ('source_manifest','sampling_schedule')}),flush=True)

if __name__=='__main__':main()

