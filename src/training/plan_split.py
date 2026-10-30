"""8:1:1 multi-label ORIGINAL-image allocation; preserve illumination pairs.

Physical scenes can cross splits by the user's 2026-10-29 training-coverage
priority. They remain visible in metadata; this is NOT independent-scene eval.
"""
import collections
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

HERE = Path(__file__).resolve().parent
FILES = {'train': 'train_originals.coco.json', 'valid': 'imx219_valid.coco.json', 'test': 'imx219_test.coco.json'}
SPLITS = list(FILES)


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def main():
    images, annotations, old = {}, {}, {}
    cats = None
    for split, filename in FILES.items():
        d = json.loads((HERE / 'source' / filename).read_text())
        cats = d['categories']
        for im in d['images']:
            assert im['source_key'] not in images
            images[im['source_key']] = im
            old[im['source_key']] = split
        for a in d['annotations']:
            assert a['id'] not in annotations
            annotations[a['id']] = a
    names = {c['id']: c['name'] for c in cats}
    by = collections.defaultdict(list)
    for a in annotations.values():
        by[a['image_id']].append(a)
    assert len(images) == 553 and len(annotations) == 5253
    units, taken = [], set()
    # Matched ordinal frames from the two ten-frame ON/OFF recordings.
    # Correspondence is a conservative grouping rule, not a measured timestamp pair.
    for start in (42, 62, 82, 102, 122, 142, 162, 182):
        for offset in range(10):
            keys = [f'imx21_{start+offset:04d}', f'imx21_{start+10+offset:04d}']
            assert all(k in images for k in keys)
            units.append(keys)
            taken.update(keys)
    units += [[k] for k in sorted(images) if k not in taken]
    assert len({images[k]['image_sha256'] for k in images}) == len(images)
    features = collections.defaultdict(lambda: np.zeros(len(units)))
    for u, keys in enumerate(units):
        for k in keys:
            im = images[k]
            counts = collections.Counter(a['category_id'] for a in by[im['id']])
            features['images'][u] += 1
            light = 'on' if 'with_light' in im['capture_session'] else ('off' if k.startswith('imx21_') else 'unrecorded')
            for cid, count in counts.items():
                features['class_images:' + names[cid]][u] += 1
                features['class_objects:' + names[cid]][u] += count
                features[f'condition:{names[cid]}:{im["range_group"]}:{light}'][u] += 1
            if not any(cid != 6 for cid in counts):
                features['hazard_negative'][u] += 1
                features[f'negative_condition:{light}'][u] += 1
            features['capture:' + im['capture_session']][u] += 1
    nu = len(units)
    # Ratio objectives include both class-positive image counts and object counts.
    targets = [(f, s, ratio) for f in features if not f.startswith('capture:')
               for s, ratio in enumerate((.8, .1, .1))]
    nx = nu * 3
    nv = nx + len(targets)
    constraints = []
    lo, hi = [], []
    def add(entries, lower, upper):
        constraints.append(dict(entries)); lo.append(lower); hi.append(upper)
    for u in range(nu):
        add({u*3+s: 1 for s in range(3)}, 1, 1)
    # All source images, disjoint and exhaustive. Largest-remainder 442/55/56.
    for s, count in enumerate((442, 55, 56)):
        add({u*3+s: float(v) for u, v in enumerate(features['images']) if v}, count, count)
    c = np.zeros(nv)
    rng = np.random.default_rng(20261029)
    c[:nx] = rng.random(nx)*.00001
    for j, (f, s, ratio) in enumerate(targets):
        vals = features[f]; total = vals.sum(); target = total*ratio
        entries = {u*3+s: float(v) for u, v in enumerate(vals) if v}
        # Hard guarantee: each class's image ratio differs by < one source image.
        if f.startswith('class_images:'):
            add(entries, np.floor(target), np.ceil(target))
        # Every recorded class/distance/light condition present in at least three
        # distinct allocation units must appear in all three splits.
        if (f.startswith(('condition:', 'negative_condition:')) or f == 'hazard_negative') and np.count_nonzero(vals) >= 3:
            add(entries, 1, np.inf)
        neg = {i: -v for i, v in entries.items()}
        entries[nx+j] = -1; neg[nx+j] = -1
        add(entries, -np.inf, target); add(neg, -np.inf, -target)
        c[nx+j] = (100 if f.startswith('class_images:') else
                   10 if f.startswith('class_objects:') else 1) / max(1, total)
    matrix = lil_matrix((len(constraints), nv))
    for row, entries in enumerate(constraints):
        for col, value in entries.items(): matrix[row, col] = value
    result = milp(c, integrality=np.r_[np.ones(nx), np.zeros(nv-nx)],
                  bounds=Bounds(np.zeros(nv), np.r_[np.ones(nx), np.full(nv-nx, np.inf)]),
                  constraints=LinearConstraint(matrix.tocsc(), lo, hi),
                  options={'time_limit': 90, 'mip_rel_gap': .005})
    if result.x is None: raise RuntimeError(result.message)
    assignment = {}
    for u, keys in enumerate(units):
        choice = np.flatnonzero(result.x[u*3:u*3+3] > .5)
        assert len(choice) == 1
        for k in keys: assignment[k] = SPLITS[int(choice[0])]
    summary = {}
    for f, vals in features.items():
        if f.startswith('capture:'): continue
        counts = {s: int(sum(vals[u] for u, keys in enumerate(units) if assignment[keys[0]] == s)) for s in SPLITS}
        summary[f] = dict(total=int(vals.sum()), counts=counts,
                          percentages={s: round(n/vals.sum()*100, 3) for s, n in counts.items()})
        if f.startswith('class_images:'):
            assert all(abs(counts[s] - vals.sum()*r) < 1.0000001 for s, r in zip(SPLITS, (.8, .1, .1)))
    groups = {}
    for field in ('physical_capture_episode', 'capture_session'):
        group_splits = collections.defaultdict(set)
        for k, im in images.items(): group_splits[im[field]].add(assignment[k])
        groups[field] = {g: sorted(ss) for g, ss in group_splits.items() if len(ss)>1}
    report = dict(result='PASS', method='multi-label constrained original-image 8:1:1 allocation',
                  solver_status=result.message, solver_gap=float(result.mip_gap),
                  exact_global_counts=[442,55,56], all_originals=553, annotations_unchanged=5253,
                  class_image_rounding_error_less_than_one=True, summary=summary,
                  units=units, allocation=assignment, previous_allocation=old,
                  on_off_ordinal_pair_count=80, source_labels_changed=False,
                  independence='shared physical scenes; frame-disjoint in-distribution diagnostics only',
                  shared_source_groups=groups,
                  authorization='User requested no empty training conditions, delegated split judgment, then explicitly required per-class 8:1:1 and best-weight retraining.',
                  source_manifest_sha256={f:hashlib.sha256((HERE/'source'/f).read_bytes()).hexdigest() for f in FILES.values()})
    write(HERE/'split_plan.json', report)
    with (HERE/'ratio_summary.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['feature','total','train','valid','test','train_pct','valid_pct','test_pct'])
        for name,row in summary.items(): writer.writerow([name,row['total'],*[row['counts'][s] for s in SPLITS],*[row['percentages'][s] for s in SPLITS]])
    print(json.dumps({k:v for k,v in report.items() if k not in ('summary','units','allocation','previous_allocation','shared_source_groups')}, indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k.startswith('class_') or k in ('images','hazard_negative')}, indent=2))


if __name__ == '__main__': main()
