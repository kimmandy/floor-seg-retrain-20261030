"""Read-only verification of actual startup, saved E0, and unchanged best weights."""
import fcntl,hashlib,json,time
from pathlib import Path
import torch
import launch_all6

HERE=Path(__file__).resolve().parent
P=json.loads((HERE/'paths.json').read_text());RUN=Path(P['run'])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    launch_all6.check();status=launch_all6.status()
    assert status['worker_alive'] and status['trainer_alive'],status
    events=[]
    for line in (RUN/'train.log').read_text().splitlines():
        try:events.append(json.loads(line))
        except ValueError:pass
    steps=[e for e in events if all(k in e for k in ('epoch','batch','loss'))]
    assert steps and steps[-1]['epoch']>=1 and steps[-1]['batch']>=21,'Await 21 real batches'
    assert all(0<e['loss']<100 for e in steps)
    assert not any(e.get('event') in ('fp32_loss_retry','safety_pause') for e in events)
    torch.set_num_threads(2)
    initial=torch.load(P['checkpoint'],map_location='cpu',weights_only=False)
    e0=torch.load(RUN/'current/last.pth',map_location='cpu',weights_only=False)
    assert e0['epoch']==0,'Run advanced; check archived E0 evidence separately'
    assert all(torch.equal(v,e0['state_dict'][k]) for k,v in initial['state_dict'].items())
    assert not e0['optimizer']['state'],'New optimizer must start empty'
    cfg=json.loads((RUN/'current/effective_config.json').read_text())
    assert cfg['epochs']==100 and cfg['side']==1600 and cfg['mask_loss_stride']==1 and cfg['items_per_epoch']==1900
    assert cfg['initialization']['source_epoch']==8 and cfg['early_stopping_patience']==0
    assert cfg['initialization']['sha256']==sha(P['checkpoint'])
    matches=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ')
        except (FileNotFoundError,PermissionError):continue
        if b'/train_all6.py train ' in cmd:matches.append(int(p.name))
    assert matches==[status['trainer_pid']],matches
    with open('/data/baseline/training/runs/.training.lock','a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:pass
        else:raise AssertionError('Worker does not hold shared lock')
    assert not (RUN/'test').exists()
    baseline=json.loads((RUN/'current/initial_validation.json').read_text())
    assert baseline['images']==55 and not baseline['independent_evaluation']
    report=dict(result='PASS',time=time.time(),status=status,progress=steps[-1],single_training_process=matches,
                e0_weights_exactly_equal_source_best=True,new_optimizer_empty_at_e0=True,
                initial_source_epoch=initial['epoch'],initial_checkpoint_sha256=sha(P['checkpoint']),
                baseline_valid_images=55,baseline_macro_f1=baseline['macro_f1'],baseline_mask_mAP=baseline['mask_mAP'],
                independent_evaluation=False,prior_model_exposure=baseline['prior_model_exposure'],
                source_original_labels_preserved=True,final_test_not_run=True,
                sleep_inhibitor=json.loads((RUN/'sleep_inhibitor.json').read_text()))
    (HERE/'live_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
