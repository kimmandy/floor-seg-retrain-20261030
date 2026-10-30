"""Initial weights (E8, epoch 0) vs current best epoch, valid55. Usage: compare_init.py <run current dir>"""
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
init = json.load(open(d / 'initial_validation.json'))
rows = [json.loads(l) for l in open(d / 'epochs.jsonl') if l.strip()]
best = max(rows, key=lambda r: r['valid']['macro_f1'])
keys = ['macro_f1', 'mask_mAP', 'mask_AP50', 'hazard_fp_per_negative_image']
cls = list(init['classes'])
print(f"{'':12}{'E8 init':>9}{'best ep'+str(best['epoch']):>10}")
for k in keys:
    print(f"{k:12}{init[k]:9.3f}{best['valid'][k]:10.3f}")
for c in cls:
    print(f"{c:12}{init['classes'][c]['f1']:9.3f}{best['valid']['classes'][c]['f1']:10.3f}")
