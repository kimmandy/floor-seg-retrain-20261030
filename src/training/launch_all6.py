"""Frozen inputs, one common lock, detached worker, explicit resume only."""
import argparse,fcntl,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/home/user/work')
P=json.loads((HERE/'paths.json').read_text())
RUN=Path(P['run'])
PY=Path('/home/user/train_env/.venv/bin/python')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def status():
    result=dict(run=str(RUN),dataset=P['dataset'],epochs=P['epochs'])
    if (RUN/'status.txt').exists():result['status']=(RUN/'status.txt').read_text().strip()
    else:result['status']='not_started'
    for name in ('worker','trainer'):
        path=RUN/(name+'.pid')
        if path.exists():
            pid=int(path.read_text());result[name+'_pid']=pid
            proc=Path('/proc')/str(pid);cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ') if proc.exists() else b''
            result[name+'_alive']=str(HERE).encode() in cmd
    if (RUN/'current/epochs.jsonl').exists():
        e=json.loads((RUN/'current/epochs.jsonl').read_text().splitlines()[-1]);result.update(completed_epoch=e['epoch'],best_selection_score=e['best_selection_score'])
    if (RUN/'train.log').exists():
        with (RUN/'train.log').open('rb') as f:
            f.seek(max(0,(RUN/'train.log').stat().st_size-16000));lines=f.read().decode(errors='replace').splitlines()
        for line in reversed(lines):
            try:e=json.loads(line)
            except ValueError:continue
            if all(k in e for k in ('epoch','batch','batches')):result['progress']=e;break
    return result

def check():
    freeze=json.loads((HERE/'launch_hashes.json').read_text())
    for name,digest in freeze.items():
        if sha(name)!=digest:raise ValueError('Frozen input changed: '+name)
    for report in ('metric_tests.json','dataset_verification.json','preflight_report.json','original_verification.json','coverage_check.json'):
        if json.loads((HERE/report).read_text())['result']!='PASS':raise ValueError('Preflight not passed: '+report)
    data=Path(P['dataset'])
    for name,digest in json.loads((data/'manifest_hashes.json').read_text()).items():
        if sha(data/name)!=digest:raise ValueError('Dataset changed: '+name)
    return dict(result='PASS',dataset=P['dataset'],train_originals=442,train_items=None,valid=55,test=56,
                initialization='preserved_previous_valid_best_weights_new_optimizer',independent_evaluation=False,physical_scenes_shared=True,epochs=100,selection='six-class valid macro F1',early_stopping=0,checkpoint_interval=10)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--status',action='store_true');ap.add_argument('--check',action='store_true');ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    if args.status:print(json.dumps(status(),indent=2));return
    report=check()
    if args.check:print(json.dumps(report,indent=2));return
    fd=os.open(ROOT/'runs/.training.lock',os.O_CREAT|os.O_RDWR,0o600)
    try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        current=status()
        if current.get('worker_alive'):print(json.dumps(current));return
        raise RuntimeError('Another worker or preflight holds the training lock')
    if RUN.exists() and not args.resume:
        os.close(fd);print(json.dumps(status()));return
    if args.resume:
        if not (RUN/'current/last.pth').exists() or status()['status']=='complete':raise ValueError('No resumable incomplete run')
    else:RUN.mkdir(parents=True,exist_ok=False)
    previous=(RUN.parent/'latest_training.txt').read_text() if (RUN.parent/'latest_training.txt').exists() else ''
    if not args.resume:(RUN/'previous_latest_training.txt').write_text(previous)
    os.dup2(fd,9);os.set_inheritable(9,True)
    if fd!=9:os.close(fd)
    (RUN/'status.txt').write_text('starting\n')
    with (RUN/'train.log').open('ab') as log:
        command=[str(PY),'-u',str(HERE/'worker_all6.py')]+(['--resume'] if args.resume else [])
        worker=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(9,))
    (RUN/'worker.pid').write_text(str(worker.pid));(RUN.parent/'latest_training.txt').write_text(str(RUN)+'\n')
    record=dict(report,worker_pid=worker.pid,run=str(RUN),previous_latest=previous.strip(),resume=args.resume,detached=True,time=time.time())
    (RUN/('resume_launch.json' if args.resume else 'launch.json')).write_text(json.dumps(record,indent=2));os.close(9)
    print(json.dumps(record),flush=True)
if __name__=='__main__':main()
