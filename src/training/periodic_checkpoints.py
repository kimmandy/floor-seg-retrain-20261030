"""Keep immutable checkpoint snapshots without interrupting an active trainer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def archive_checkpoint(checkpoint, epoch, interval=10, output_dir=None):
    if interval <= 0:
        raise ValueError('interval must be positive')
    if epoch <= 0 or epoch % interval:
        return None
    checkpoint = Path(checkpoint)
    target = Path(output_dir or checkpoint.parent) / f'epoch_{epoch:03d}.pth'
    # last.pth is always replaced atomically by train.save; its old inode is immutable.
    try:
        os.link(checkpoint, target)
        event = 'periodic_checkpoint_saved'
    except FileExistsError:
        if digest(checkpoint) != digest(target):
            raise RuntimeError(f'Existing periodic checkpoint differs: {target}')
        event = 'periodic_checkpoint_already_saved'
    print(json.dumps(dict(event=event, epoch=epoch, path=str(target))), flush=True)
    return target


def observe_checkpoint(checkpoint, interval=10):
    import torch
    checkpoint = Path(checkpoint)
    # Pin one atomically published checkpoint while the trainer may replace last.pth.
    with tempfile.TemporaryDirectory(prefix='.periodic_', dir=checkpoint.parent) as directory:
        pinned = Path(directory) / 'last.pth'
        os.link(checkpoint, pinned)
        state = torch.load(pinned, map_location='cpu', weights_only=False)
        epoch = int(state['epoch'])
        del state
        if epoch <= 0 or epoch % interval:
            return None
        target = archive_checkpoint(pinned, epoch, interval, checkpoint.parent)
        print(json.dumps(dict(event='active_run_checkpoint_preserved', epoch=epoch,
                              path=str(target), sha256=digest(target))), flush=True)
        return target


def watch(checkpoint, trainer_pid, interval):
    import fcntl
    checkpoint = Path(checkpoint)
    with (checkpoint.parent / '.periodic_checkpoint.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(json.dumps(dict(event='periodic_watcher_started', trainer_pid=trainer_pid,
                              interval=interval, checkpoint=str(checkpoint))), flush=True)
        previous = None
        while True:
            try:
                os.kill(trainer_pid, 0)
                alive = True
            except ProcessLookupError:
                alive = False
            if checkpoint.exists():
                st = checkpoint.stat()
                current = (st.st_ino, st.st_mtime_ns, st.st_size)
                if current != previous:
                    observe_checkpoint(checkpoint, interval)
                    previous = current
            if not alive:
                print(json.dumps(dict(event='periodic_watcher_finished')), flush=True)
                return
            time.sleep(5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--trainer-pid', type=int, required=True)
    parser.add_argument('--interval', type=int, default=10)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('--interval must be positive')
    watch(args.checkpoint, args.trainer_pid, args.interval)
