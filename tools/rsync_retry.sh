#!/bin/bash
# Restart chunks whose rc != 0, staggered (AGX sshd drops bursts of new connections), retry until success.
# Usage: rsync_retry.sh <chunk ids...>
SRC=/data/work/home_train_20261029
DST=$HOME/home_train_20261029
W=$HOME/rsync_par
for i in "$@"; do
  (
    for try in 1 2 3 4 5 6 7 8 9 10; do
      rsync -aL --partial --files-from="$W/chunk.$i" "user@<TRAIN_SERVER>:$SRC/" "$DST/" >> "$W/$i.log" 2>&1
      rc=$?; echo $rc > "$W/$i.rc"
      [ $rc -eq 0 ] && break
      sleep $((5 + RANDOM % 10))
    done
  ) &
  sleep 2
done
wait
