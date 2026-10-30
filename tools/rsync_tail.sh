#!/bin/bash
# Tail speed-up: for each still-running chunk, add a helper rsync that walks the chunk list in reverse.
# Already-complete files are skipped by rsync; final full rsync verifies everything.
SRC=/data/work/home_train_20261029
DST=$HOME/home_train_20261029
W=$HOME/rsync_par
ids=$(ps -eo args | grep '^rsync -aL --partial --files-from=' | grep -oE 'chunk\.[0-9]+ ' | grep -oE '[0-9]+' | sort -u)
for i in $ids; do
  tac "$W/chunk.$i" > "$W/rev.$i"
  ( rsync -aL --files-from="$W/rev.$i" "user@<TRAIN_SERVER>:$SRC/" "$DST/" > "$W/rev.$i.log" 2>&1; echo $? > "$W/rev.$i.rcx" ) &
  sleep 2
done
wait
