#!/bin/bash
# Stop the run (VRAM spilled into Windows shared sysmem -> 12x slowdown) and resume from last.pth
# with a fragmentation-resistant allocator. Training files are not modified.
set -u
cd /home/user/work
R=runs/ratio811_fix_20261029
W=$(cat $R/worker.pid); T=$(cat $R/trainer.pid)
echo "worker=$W trainer=$T"
ls -la --time-style=+%H:%M $R/current/last.pth
kill $T; sleep 15; kill $W 2>/dev/null; sleep 5
ps -p $W,$T -o pid=,args= | cut -c1-80
echo "status: $(cat $R/status.txt)"
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# same as start_training.sh --resume, but allocator conf replaced
export PATH="/home/user/train_env/.venv/bin:/usr/local/cuda/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6
export PYTHONPATH="/home/user/work/setup"
/home/user/train_env/.venv/bin/python setup/launch_all6.py --resume 2>&1 | tail -2 | cut -c1-300
sleep 5
echo "env of new trainer: $(tr '\0' '\n' < /proc/$(cat $R/trainer.pid)/environ | grep PYTORCH_CUDA)"
