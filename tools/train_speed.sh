#!/bin/bash
# Per-batch timing trend for the current epoch + GPU power/clock + host memory
L=$HOME/home_train_20261029/runs/ratio811_fix_20261029/train.log
python3 - "$L" <<'EOF'
import json,sys
rows=[]
for line in open(sys.argv[1]):
    if line.startswith('{"epoch"') and '"batch"' in line:
        try: rows.append(json.loads(line))
        except: pass
ep=rows[-1]['epoch']; cur=[r for r in rows if r['epoch']==ep]
prev=[r for r in rows if r['epoch']==ep-1]
def per10(rs): return [(rs[i]['batch'], round(rs[i]['seconds']-rs[i-1]['seconds'],1)) for i in range(1,len(rs))]
if prev: p=per10(prev); print('prev epoch s/10 batches median', sorted(x for _,x in p)[len(p)//2])
c=per10(cur); print('epoch',ep,'s/10 batches:', c[::max(1,len(c)//15)])
EOF
for i in 1 2 3; do nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm --format=csv,noheader; sleep 2; done
free -g | head -2
