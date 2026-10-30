#!/bin/bash
# Download progress: bytes now vs N seconds later, done files, chunk return codes
N="${1:-60}"
D=$HOME/home_train_20261029/data/files
sz() { find "$D" -type f -printf '%s\n' | awk '{t+=$1} END {print t+0}'; }
a=$(sz); sleep "$N"; b=$(sz)
rate=$(( (b - a) / N / 1024 ))
done_n=$(find "$D" -type f ! -name '.*' | wc -l)
total=27000   # MB, approx
echo "rate ${rate} KB/s | got $((b / 1048576)) MB / ~${total} MB | files ${done_n}/4809"
[ "$rate" -gt 0 ] && echo "ETA ~$(( (total * 1024 - b / 1024) / rate / 3600 )) h"
echo "active rsync chunks: $(pgrep -f 'rsync -aL --partial --files-from' | xargs -r ps -o ppid= -p | sort -u | wc -l)"
echo "rc: $(for f in $HOME/rsync_par/*.rc; do printf '%s=%s ' "$(basename "$f" .rc)" "$(cat "$f")"; done)"
