"""Source-balanced full/crop exposure, preserving measured distance labels."""
from collections import Counter, defaultdict
import random

POLICY = 'all6_811_20261029_condition_covered_full_crop_crowd_kmj_v1'
COUNTS = {
    'fastener_near':80, 'fastener_mid':120, 'fastener_far':160, 'fastener_moving':120,
    'imx27_bolt_button':160, 'imx27_nut_button':120,
    'imx27_nut_near':40, 'imx27_nut_mid':80, 'imx27_nut_far':120,
    'imx28_m3_nut_mid':160, 'imx28_m3_bolt':160,
    'liquid_mixed_near':80, 'liquid_mixed_mid':80, 'liquid_mixed_far':80,
    'liquid_water_moving':80, 'liquid_water_unknown':20, 'liquid_oil':80, 'liquid_paint':80, 'empty':80,
    'fastener_crowd':280,  # 2026-10-29 kmj: crowd_paste 밀집 장면
}

def bucket(im, annotations):
    positive = {a['category_id'] for a in annotations} & {1,2,3,4,5}
    if not positive: return 'empty'
    if im['source_key'].startswith('imx21_') and positive & {1,4,5}:
        return 'liquid_mixed_'+im['range_group']
    if 1 in positive: return 'liquid_water_'+('moving' if im['range_group']=='moving' else 'unknown')
    for cid, label in ((4,'oil'),(5,'paint')):
        if cid in positive: return 'liquid_' + label
    result = im['sampling_bucket']
    if result not in COUNTS: raise ValueError(('unplanned metal bucket', im['source_key'],result))
    return result

def epoch_images(images, epoch):
    pools = defaultdict(lambda: defaultdict(list))
    for im in images:
        pools[(im['sampling_bucket'], im['sample_view'])][im['source_key']].append(im)
    if {b for b,v in pools} != set(COUNTS): raise ValueError('Sampling bucket missing or unexpected')
    chosen = []
    for b,count in COUNTS.items():
        for view in ('full','crop'):
            sources=pools[(b,view)]; keys=sorted(sources)
            if count//2 < len(keys): raise ValueError(('Quota would omit sources',b,view))
            random.Random(20261028).shuffle(keys)
            for j in range(count//2):
                key=keys[(epoch*(count//2)+j)%len(keys)]
                variants=sorted(sources[key],key=lambda im:im['id'])
                chosen.append(variants[(epoch+j//len(keys))%len(variants)])
    random.Random(20261028+epoch).shuffle(chosen)
    assert Counter(i['sampling_bucket'] for i in chosen)==COUNTS
    assert Counter(i['sample_view'] for i in chosen)=={'full':sum(COUNTS.values())//2,'crop':sum(COUNTS.values())//2}
    assert {i['source_key'] for i in chosen}=={i['source_key'] for i in images}
    return chosen
