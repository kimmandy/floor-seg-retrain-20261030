"""Per-epoch valid summary: macro F1, per-class F1, mAP-like keys, best marker. Usage: epoch_table.py <epochs.jsonl>"""
import json, sys

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
v0 = rows[0]['valid']
scalar_keys = [k for k, x in v0.items() if isinstance(x, (int, float)) and k not in ('epoch', 'images', 'negative_images')]
classes = list(v0['classes'])
print('valid scalar keys:', scalar_keys)

def macro(v):
    for k in ('macro_f1', 'six_class_macro_f1', 'macro_f1_6'):
        if k in v: return v[k]
    f = [c['f1'] for c in v['classes'].values()]
    return sum(f) / len(f)

best = -1
hdr = 'ep  loss   macroF1 ' + ' '.join(f'{c[:6]:>6}' for c in classes) + '  hazFP'
print(hdr)
for r in rows:
    v = r['valid']; m = macro(v)
    mark = ' *best' if m > best else ''
    best = max(best, m)
    print(f"{r['epoch']:>2}  {r.get('mean_loss', 0):.3f}  {m:.3f}  " +
          ' '.join(f"{v['classes'][c]['f1']:6.3f}" for c in classes) +
          f"  {v.get('hazard_fp_per_negative_image', float('nan')):.2f}{mark}")
