"""Bind only the locally verified final code, manifests, and source checkpoint."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
p=json.loads((HERE/'paths.json').read_text());data=Path(p['dataset'])
reports=['metric_tests.json','dataset_verification.json','preflight_report.json','original_verification.json','coverage_check.json']
for n in reports:assert json.loads((HERE/n).read_text())['result']=='PASS',n
files=list(HERE.glob('*.py'))+[HERE/'paths.json',HERE/'start_training.sh',HERE/'initialization.json']+[HERE/n for n in reports]
files += [data/'manifest_hashes.json',Path(p['source_dataset'])/'manifest_hashes.json',Path(p['checkpoint']),Path('/data/baseline/training/env.sh')]
lock=Path('/data/baseline/training/requirements.lock.txt')
if lock.exists():files.append(lock)
hashes={str(f):sha(f) for f in files}
(HERE/'launch_hashes.json').write_text(json.dumps(hashes,indent=2))
print(json.dumps(dict(result='PASS',frozen_files=len(hashes),checkpoint=str(p['checkpoint']))))
