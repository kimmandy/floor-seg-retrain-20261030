"""Preserve exact completed checkpoints, then stop only the identified old worker."""
import fcntl, hashlib, json, os, shutil, signal, time
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent
OLD=Path('/data/baseline/training/runs/imx219_all6_coco_20261028_1948')
P=json.loads((HERE/'paths.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')

def main():
    worker=int((OLD/'worker.pid').read_text());trainer=int((OLD/'trainer.pid').read_text())
    for pid,name in [(worker,'worker_all6.py'),(trainer,'train_all6.py')]:
        command=(Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        assert '/setup_20261028/speed_review_2208/' in command and name in command,command
    (HERE/'before').mkdir(exist_ok=False)
    for name in ['best','last']:
        src=OLD/'current'/(name+'.pth');dst=HERE/'before'/(name+'.pth')
        for attempt in range(3):
            before=sha(src);shutil.copy2(src,dst)
            if before==sha(src)==sha(dst):break
        else:raise RuntimeError('Checkpoint changed during snapshot')
    torch.set_num_threads(2)
    best=torch.load(HERE/'before/best.pth',map_location='cpu',weights_only=False)
    last=torch.load(HERE/'before/last.pth',map_location='cpu',weights_only=False)
    assert list(best['meta']['dataset_meta']['classes'])==['water','bolt','nut','oil','paint','reflection']
    events=[json.loads(l) for l in (OLD/'current/epochs.jsonl').read_text().splitlines()]
    event=next(e for e in events if e['epoch']==best['epoch'])
    assert abs(event['valid']['macro_f1']-best['best'])<1e-12
    initial=Path(P['checkpoint']);assert not initial.exists();shutil.copy2(HERE/'before/best.pth',initial)
    init=dict(checkpoint=str(initial),sha256=sha(initial),source_run=str(OLD),source_epoch=best['epoch'],
              source_valid_macro_f1=best['best'],source_metric='previous six-class valid macro F1',
              model_weights_only=True,new_optimizer=True,new_scaler=True,new_schedule_epochs=100,
              source_last_epoch=last['epoch'],old_dataset=str(Path('/data/baseline/datasets/imx219_all6_aug_20261028_1948')),
              inherited_data_exposure=True,independent_evaluation=False)
    write(HERE/'initialization.json',init)
    shutil.copy2('/data/baseline/training/start_training.sh',HERE/'before/start_training.sh')
    write(HERE/'handoff_intent.json',dict(worker=worker,trainer=trainer,initialization=init,time=time.time(),reason='User requested per-class 8:1:1 and new training from current best; preserve last completed epoch, abandon unsaved old-dataset batch progress.'))
    os.kill(worker,signal.SIGTERM)
    for _ in range(100):
        active=[]
        for pid in (worker,trainer):
            f=Path('/proc')/str(pid)/'stat'
            if f.exists() and f.read_text().split()[2]!='Z':active.append(pid)
        if not active:break
        time.sleep(.3)
    else:raise RuntimeError('Old processes still active; do not launch another trainer')
    with open('/data/baseline/training/runs/.training.lock','a') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    record=dict(result='PASS',old_worker_stopped=worker,old_trainer_stopped=trainer,training_lock_available=True,initialization=init,
                original_best_still_exists=(OLD/'current/best.pth').exists(),original_last_still_exists=(OLD/'current/last.pth').exists(),time=time.time())
    write(HERE/'handoff_verification.json',record);print(json.dumps(record),flush=True)

if __name__=='__main__':main()
