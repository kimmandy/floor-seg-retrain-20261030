#!/bin/bash
# Parallel AGX -> WSL copy: per-connection speed is capped (~95 KB/s), so split the file list over N rsyncs.
# Usage: agx_rsync_par.sh [N=24]   log: ~/rsync_par/*.log, done marker: ~/rsync_par/ALL_DONE
N="${1:-24}"
export SSH_ASKPASS=${SSH_ASKPASS:-} SSH_ASKPASS_REQUIRE=force DISPLAY=dummy
SRC=/data/work/home_train_20261029
DST=$HOME/home_train_20261029
W=$HOME/rsync_par
mkdir -p "$W" "$DST"
rm -f "$W"/ALL_DONE "$W"/chunk.* "$W"/*.log "$W"/*.rc

# small stuff first (setup, checkpoint, json, iou50 setup folder)
rsync -aL --exclude 'data/files/' "user@<TRAIN_SERVER>:$SRC/" "$DST/" > "$W/small.log" 2>&1
rsync -a user@<TRAIN_SERVER>:/data/work/handoff_20261028_iou50/ "$HOME/handoff_20261028_iou50/" >> "$W/small.log" 2>&1

# big files: sorted list, round-robin into N chunks so sizes even out
ssh user@<TRAIN_SERVER> "cd $SRC && find -L data/files -type f" | sort > "$W/all_files.txt"
awk -v n="$N" -v w="$W" '{ print > (w "/chunk." (NR % n)) }' "$W/all_files.txt"

for c in "$W"/chunk.*; do
  i="${c##*.}"
  ( rsync -aL --partial --files-from="$c" "user@<TRAIN_SERVER>:$SRC/" "$DST/" > "$W/$i.log" 2>&1; echo $? > "$W/$i.rc" ) &
done
wait
fails=$(grep -Lx 0 "$W"/*.rc | wc -l)
echo "chunks=$(ls "$W"/*.rc | wc -l) failed=$fails" > "$W/ALL_DONE"
