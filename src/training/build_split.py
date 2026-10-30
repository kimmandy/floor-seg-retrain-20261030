"""Materialize reviewed originals using an audited per-class 8:1:1 plan."""
import collections, copy, csv, hashlib, json, shutil, time
from pathlib import Path

HERE=Path(__file__).resolve().parent
P=json.loads((HERE/'paths.json').read_text())
SOURCE=Path('/data/baseline/datasets/imx219_all6_split_20261028_1922')
FILES={'train':'train_originals.coco.json','valid':'imx219_valid.coco.json','test':'imx219_test.coco.json'}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj): p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')

def main():
    plan=json.loads((HERE/'split_plan.json').read_text());out=Path(P['source_dataset'])
    assert plan['result']=='PASS' and not out.exists()
    source_hashes=json.loads((SOURCE/'manifest_hashes.json').read_text())
    for f,h in source_hashes.items(): assert sha(SOURCE/f)==h,f
    images={};annotations={};by=collections.defaultdict(list)
    for split,f in FILES.items():
        assert sha(SOURCE/f)==plan['source_manifest_sha256'][f]
        d=json.loads((SOURCE/f).read_text());cats=d['categories']
        for i in d['images']:images[i['source_key']]=i
        for a in d['annotations']:annotations[a['id']]=a;by[a['image_id']].append(a)
    assert set(images)==set(plan['allocation']) and len(annotations)==5253
    out.mkdir();(out/'images').mkdir();(out/'provenance').mkdir()
    write(out/'provenance/source_manifest_hashes.json',source_hashes)
    shutil.copy2(HERE/'split_plan.json',out/'split_plan.json')
    units={key:f'ratio811_unit_{n:04d}' for n,keys in enumerate(plan['units']) for key in keys}
    new={s:dict(images=[],annotations=[],categories=cats) for s in FILES};rows=[]
    for n,(key,orig) in enumerate(sorted(images.items()),1):
        split=plan['allocation'][key];im=copy.deepcopy(orig)
        path=Path(im['file_name']);assert sha(path)==im['image_sha256'],key
        dest=out/'images'/path.name;dest.symlink_to(path)
        assert sha(dest)==im['image_sha256']
        im.update(file_name=str(dest),previous_dataset_file_name=str(path),previous_split=orig['split'],split=split,
                  split_reason='user_requested_per_class_8_1_1_coverage_priority',allocation_unit=units[key],
                  evaluation_independence='shared_physical_scene',
                  prior_model_exposure={'train':'training','valid':'checkpoint_selection','test':'not_used_for_training_or_selection'}[orig['split']])
        new[split]['images'].append(im);new[split]['annotations'].extend(copy.deepcopy(by[im['id']]))
        rows.append({k:im[k] for k in ('source_key','previous_split','split','range_group','capture_session','physical_capture_episode','allocation_unit','image_sha256','prior_model_exposure')})
        if n%100==0:print(json.dumps(dict(event='verified_originals',done=n,total=553)),flush=True)
    assert {a['id']:a for d in new.values() for a in d['annotations']}==annotations
    names={c['id']:c['name'] for c in cats};summary={}
    for s,d in new.items():
        counts=collections.Counter(a['category_id'] for a in d['annotations'])
        positive={name:sum(any(a['category_id']==cid for a in by[i['id']]) for i in d['images']) for cid,name in names.items()}
        assert set(counts)==set(names)
        for name in names.values():
            assert positive[name]==plan['summary']['class_images:'+name]['counts'][s]
            assert counts[next(c for c,n in names.items() if n==name)]==plan['summary']['class_objects:'+name]['counts'][s]
        summary[s]=dict(images=len(d['images']),positive_images=positive,instances={names[c]:v for c,v in counts.items()},
                        prior_model_exposure=dict(collections.Counter(i['prior_model_exposure'] for i in d['images'])),
                        negative_images=sum(not any(a['category_id']!=6 for a in by[i['id']]) for i in d['images']))
        write(out/FILES[s],d);(out/(s+'set')/'images').mkdir(parents=True)
        (out/(s+'set')/'annotations.coco.json').symlink_to('../'+FILES[s])
        for im in d['images']:(out/(s+'set')/'images'/Path(im['file_name']).name).symlink_to('../../images/'+Path(im['file_name']).name)
    for left,right in [('train','valid'),('train','test'),('valid','test')]:
        for field in ('source_key','image_sha256','allocation_unit'):
            assert not {i[field] for i in new[left]['images']}&{i[field] for i in new[right]['images']},(left,right,field)
    oldpolicy=json.loads((SOURCE/'evaluation_policy.json').read_text())
    policy=copy.deepcopy(oldpolicy)
    policy.update(version='imx219_per_class_811_shared_scene_20261029_v1',required_initialization='user_requested_previous_valid_best_weights',
        allow_shared_physical_scenes=True,independent_evaluation=False,
        allocation_ratio=[8,1,1],allocation_unit='original_image; ordinal ON/OFF pairs kept together',
        shared_group_authorization=plan['authorization'],split_guard_fields=['source_key','image_sha256','allocation_unit'],
        training_source_keys=sorted(i['source_key'] for i in new['train']['images']),
        training_groups=sorted({i['physical_capture_episode'] for i in new['train']['images']}),
        training_allocation_units=sorted({i['allocation_unit'] for i in new['train']['images']}),
        training_hashes=sorted(i['image_sha256'] for i in new['train']['images']),
        source_capture_session_overlap=plan['shared_source_groups']['capture_session'],
        physical_episode_overlap=plan['shared_source_groups']['physical_capture_episode'],
        prior_model_exposure={s:summary[s]['prior_model_exposure'] for s in FILES},
        limitations=['Frame-disjoint 8:1:1 condition coverage; physical scenes and adjacent frames can cross splits. Scores can be optimistic and are not independent-scene generalization.',
                     'Requested previous-best initialization has trained on some newly held-out original images; those exposures are reported per frame and split. Even frame independence from model history is not claimed.',
                     'Unknown distances remain unknown. There are no confirmed far-oil originals, so no far-oil capability is invented.',
                     'Class counts are nearest feasible integer 8:1:1 ratios, not augmented item ratios. Test never selects best.'],
        splits={s:dict(images=len(new[s]['images']),expected_frames={i['source_key']:{k:i[k] for k in ('image_sha256','group','capture_session','physical_capture_episode','allocation_unit','annotation_revision','prior_model_exposure')} for i in new[s]['images']}) for s in ('valid','test')})
    write(out/'evaluation_policy.json',policy)
    with (out/'split_manifest.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report=dict(result='PASS',time=time.time(),source=str(SOURCE),output=str(out),summary=summary,
                originals_sha_verified=553,unchanged_annotations=5253,source_manifest_unchanged=True,
                image_hash_and_allocation_unit_overlap=0,physical_scenes_shared=True,independent_evaluation=False,
                web_labels_modified=False,source_files_modified=False,training_started=False)
    write(out/'build_report.json',report)
    for f,h in source_hashes.items():assert sha(SOURCE/f)==h
    files=list(FILES.values())+['split_plan.json','split_manifest.csv','evaluation_policy.json','build_report.json']
    write(out/'manifest_hashes.json',{f:sha(out/f) for f in files})
    write(HERE/'original_verification.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
