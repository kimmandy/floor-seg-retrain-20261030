"""Build public-safe results for the repo: aggregate metrics only (no per-image data), plots, comparison table.

Usage: make_report.py <ours epochs.jsonl> <ours initial_validation.json> <bok epochs.jsonl> <out dir>
"""
import json, sys, pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ours_p, init_p, bok_p, out = map(pathlib.Path, sys.argv[1:5])
out.mkdir(parents=True, exist_ok=True)
CLASSES = ['water', 'bolt', 'nut', 'oil', 'paint', 'reflection']


def slim(v):
    """Aggregate-only view of one validation record."""
    coco = v.get('coco_mask_ap', {}).get('all', {})
    return dict(
        macro_f1=v['macro_f1'], mask_mAP=v['mask_mAP'], mask_AP50=v['mask_AP50'],
        AP_small=coco.get('AP_small'), AP_medium=coco.get('AP_medium'), AP_large=coco.get('AP_large'),
        hazard_fp_per_negative_image=v['hazard_fp_per_negative_image'],
        f1={c: v['classes'][c]['f1'] for c in CLASSES},
        precision={c: v['classes'][c]['precision'] for c in CLASSES},
        recall={c: v['classes'][c]['recall'] for c in CLASSES},
        class_AP={c: coco.get('classes', {}).get(c, {}).get('AP') for c in CLASSES},
        class_AP50={c: coco.get('classes', {}).get(c, {}).get('AP50') for c in CLASSES},
    )


def load(p):
    rows = [json.loads(l) for l in open(p) if l.strip()]
    return [dict(epoch=r['epoch'], mean_loss=r['mean_loss'], valid=slim(r['valid'])) for r in rows]


ours, bok = load(ours_p), load(bok_p)
init = dict(epoch=0, valid=slim(json.load(open(init_p))))
for name, rows in (('ours_valid_epochs', ours), ('bok_ratio811_valid_epochs', bok)):
    with open(out / f'{name}.jsonl', 'w') as f:
        for r in rows: f.write(json.dumps(r) + '\n')
json.dump(init, open(out / 'initial_E8_valid.json', 'w'), indent=1)

# ---- plots: ours vs bok, same epochs
common = min(len(ours), len(bok))
ep = [r['epoch'] for r in ours[:common]]
def series(rows, f): return [f(r['valid']) for r in rows[:common]]
panels = [
    ('6-class macro F1', lambda v: v['macro_f1']),
    ('nut F1', lambda v: v['f1']['nut']),
    ('bolt F1', lambda v: v['f1']['bolt']),
    ('mask AP_small (COCO, < 32² px)', lambda v: v['AP_small']),
    ('mask AP50', lambda v: v['mask_AP50']),
    ('water F1', lambda v: v['f1']['water']),
]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, (title, f) in zip(axes.flat, panels):
    ax.plot(ep, series(bok, f), color='#9aa0a6', lw=1.6, label='baseline (bok ratio811)')
    ax.plot(ep, series(ours, f), color='#1a73e8', lw=1.8, label='this run (B1+B2+C)')
    ax.axhline(f(init['valid']), color='#d93025', ls='--', lw=1, label='E8 init')
    ax.set_title(title); ax.set_xlabel('epoch'); ax.grid(alpha=.3)
axes.flat[0].legend(loc='lower right', fontsize=9)
fig.tight_layout(); fig.savefig(out / 'valid_curves.png', dpi=120); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot([r['epoch'] for r in bok], [r['mean_loss'] for r in bok], color='#9aa0a6', label='baseline (bok ratio811)')
ax.plot([r['epoch'] for r in ours], [r['mean_loss'] for r in ours], color='#1a73e8', label='this run')
ax.set_xlabel('epoch'); ax.set_ylabel('mean train loss'); ax.grid(alpha=.3); ax.legend()
ax.set_title('Train loss (not comparable in absolute value: different box targets/data)')
fig.tight_layout(); fig.savefig(out / 'train_loss.png', dpi=120); plt.close(fig)

# ---- summary table (markdown)
def best(rows): return max(rows[:common], key=lambda r: r['valid']['macro_f1'])
def mean_last(rows, f, n=10):
    xs = [f(r['valid']) for r in rows[common - n:common]]; return sum(xs) / len(xs)
bo, bb = best(ours), best(bok)
metrics = [
    ('macro F1', lambda v: v['macro_f1']), ('mask mAP', lambda v: v['mask_mAP']),
    ('mask AP50', lambda v: v['mask_AP50']), ('**mask AP_small**', lambda v: v['AP_small']),
    ('mask AP_medium', lambda v: v['AP_medium']),
] + [(f'{c} F1', (lambda c: lambda v: v['f1'][c])(c)) for c in CLASSES] + [
    ('hazard FP / empty image', lambda v: v['hazard_fp_per_negative_image'])]
lines = [f'Valid 55 images. Epochs compared: 1–{common} (both runs).', '',
         f'| metric | E8 init | baseline best (ep {bb["epoch"]}) | **this run best (ep {bo["epoch"]})** '
         f'| baseline mean ep {common-9}–{common} | this run mean ep {common-9}–{common} |',
         '|---|---|---|---|---|---|']
fmt = lambda x: '–' if x is None else f'{x:.3f}'
for name, f in metrics:
    lines.append(f'| {name} | {fmt(f(init["valid"]))} | {fmt(f(bb["valid"]))} | **{fmt(f(bo["valid"]))}** '
                 f'| {fmt(mean_last(bok, f))} | {fmt(mean_last(ours, f))} |')
(out / 'summary_table.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
