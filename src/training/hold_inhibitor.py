"""End the temporary sleep inhibitor when this exact worker process ends."""
import sys,time
from pathlib import Path
pid,start=sys.argv[1:];p=Path('/proc')/pid/'stat'
while p.exists():
    try:
        fields=p.read_text().split()
        if fields[21]!=start or fields[2]=='Z':break
    except FileNotFoundError:break
    time.sleep(5)
