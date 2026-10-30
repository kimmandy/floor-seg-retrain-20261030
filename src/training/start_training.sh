#!/usr/bin/env bash
set -euo pipefail
ALL6_SETUP=/home/user/work/setup
ALL6_TRAIN=/home/user/train_env/.venv/..
export PATH="/home/user/train_env/.venv/bin:/usr/local/cuda/bin:$PATH" PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 PYTHONPATH="/home/user/work/setup${PYTHONPATH:+:$PYTHONPATH}"
case "${1:-}" in
  --status|--check) exec "/home/user/train_env/.venv/bin/python" "$ALL6_SETUP/launch_all6.py" "$1" ;;
  --metrics) exec "/home/user/train_env/.venv/bin/python" "$ALL6_SETUP/show_metrics.py" ;;
  --metrics-live) exec watch -n 5 "/home/user/train_env/.venv/bin/python" "$ALL6_SETUP/show_metrics.py" ;;
  --metrics-test) exec "/home/user/train_env/.venv/bin/python" "$ALL6_SETUP/show_metrics.py" --test ;;
  --detach) exec "/home/user/train_env/.venv/bin/python" "$ALL6_SETUP/launch_all6.py" ;;
  --resume) "/home/user/train_env/.venv/bin/python" "$ALL6_SETUP/launch_all6.py" --resume ;;
  --logs) ;;
  '') "/home/user/train_env/.venv/bin/python" "$ALL6_SETUP/launch_all6.py" ;;
  *) echo '사용법: start_training.sh [--status|--check|--detach|--logs|--resume|--metrics|--metrics-live|--metrics-test]'; exit 2 ;;
esac
ALL6_RUN=/home/user/work/runs/ratio811_fix_20261029
if [[ ! -f "$ALL6_RUN/worker.pid" ]]; then echo '아직 시작되지 않았습니다.'; exit 0; fi
ALL6_PID="$(cat "$ALL6_RUN/worker.pid")"
echo 'Ctrl+C는 로그 보기만 종료합니다. 학습은 계속됩니다.'
tail --pid="$ALL6_PID" -n 20 -F "$ALL6_RUN/train.log"
