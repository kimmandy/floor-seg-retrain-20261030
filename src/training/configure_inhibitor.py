"""Temporary root sleep/idle inhibitor, limited to the exact launched worker lifetime."""
import json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;p=json.loads((HERE/'paths.json').read_text());run=Path(p['run'])
pid=(run/'worker.pid').read_text().strip();start=(Path('/proc')/pid/'stat').read_text().split()[21]
unit='bok-ratio811-20261029-0821-inhibit'
command=['sudo','-S','-p','','systemd-run','--quiet','--unit='+unit,
    '/usr/bin/systemd-inhibit','--what=sleep:idle','--mode=block','--who=bok-all6',
    '--why=IMX219 six-class training worker '+pid,
    '/usr/bin/python3',str(HERE/'hold_inhibitor.py'),pid,start]
result=subprocess.run(command,input=sys.stdin.readline(),capture_output=True,text=True)
if result.returncode:raise RuntimeError(result.stderr)
status=subprocess.run(['systemctl','is-active',unit],capture_output=True,text=True)
if status.returncode:raise RuntimeError('Temporary inhibitor not active: '+status.stdout+status.stderr)
record=dict(unit=unit,worker_pid=int(pid),worker_start_time=start,status=status.stdout.strip(),permanent_power_settings_changed=False)
(run/'sleep_inhibitor.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
