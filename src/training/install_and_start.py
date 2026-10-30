"""Install only after frozen checks pass and the previous launcher is unchanged."""
import hashlib,json,os,shutil,subprocess
from pathlib import Path
import launch_all6 as launch

HERE=Path(__file__).resolve().parent
ROOT=Path('/data/baseline/training')
P=json.loads((HERE/'paths.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    report=launch.check()
    current=ROOT/'start_training.sh'
    assert sha(current)==sha(HERE/'before/start_training.sh'),'Launcher changed by another session; inspect before replacing'
    assert not Path(P['run']).exists(),'Refuse to reuse an existing run'
    temp=ROOT/'start_training_ratio811_20261029.tmp'
    assert not temp.exists();shutil.copy2(HERE/'start_training.sh',temp);temp.chmod(0o755);temp.replace(current)
    assert sha(current)==sha(HERE/'start_training.sh')
    result=subprocess.run([str(ROOT/'.venv/bin/python'),str(HERE/'launch_all6.py')],check=True,capture_output=True,text=True)
    print(result.stdout,flush=True)
    (HERE/'launcher_installation.json').write_text(json.dumps(dict(result='PASS',launcher=str(current),sha256=sha(current),old_launcher=str(HERE/'before/start_training.sh'),launch=json.loads(result.stdout)),indent=2))
if __name__=='__main__':main()
