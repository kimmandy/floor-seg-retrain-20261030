"""Own the shared lock until training and the one final heldout test end."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
P=json.loads((HERE/'paths.json').read_text());RUN=Path(P['run']);PY='/home/user/train_env/.venv/bin/python'
CHILD=None
def terminate(signum,frame):
    if CHILD is not None and CHILD.poll() is None:CHILD.terminate()
    raise InterruptedError('worker interrupted')
def execute(args):
    global CHILD
    CHILD=subprocess.Popen([PY,'-u',str(HERE/'train_all6.py')]+args,pass_fds=(9,))
    (RUN/'trainer.pid').write_text(str(CHILD.pid));code=CHILD.wait();CHILD=None
    if code:raise RuntimeError('Trainer/evaluator exited '+str(code))
def main():
    os.fstat(9);signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    try:
        (RUN/'status.txt').write_text('running\n')
        args=['train','--dataset',P['dataset'],'--epochs',str(P['epochs']),'--output',str(RUN/'current')]
        if '--resume' in sys.argv:args+=['--resume',str(RUN/'current/last.pth')]
        execute(args)
        (RUN/'status.txt').write_text('evaluating_final_test\n')
        test_out=RUN/('test_attempt_'+str(time.time_ns()))
        execute(['evaluate','--dataset',P['dataset'],'--split','test','--checkpoint',str(RUN/'current/best.pth'),'--output',str(test_out)])
        if (RUN/'test').exists():raise RuntimeError('Final test already exists; preserve both results')
        test_out.rename(RUN/'test')
        (RUN/'status.txt').write_text('complete\n')
    except BaseException as error:
        (RUN/'status.txt').write_text('interrupted\n' if isinstance(error,InterruptedError) else 'failed\n')
        (RUN/'failure.json').write_text(json.dumps(dict(error=repr(error),time=time.time())))
        if CHILD is not None and CHILD.poll() is None:
            CHILD.terminate()
            try:CHILD.wait(timeout=20)
            except subprocess.TimeoutExpired:CHILD.kill();CHILD.wait()
        raise
if __name__=='__main__':main()
