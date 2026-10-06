"""Return completed Jar base outputs to node A; publish a receipt last."""
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
source=ROOT/'artifacts/domain-evaluation-20261006/jar-base'
progress=json.loads((source/'progress.json').read_text())
if progress.get('status')!='finished' or progress.get('planned')!=26:
    raise RuntimeError('Refuse to publish incomplete Jar base sweep')
transport='ssh -p 15019 -i /workspace/bootstrap-copy/id_result -o BatchMode=yes -o ConnectTimeout=20'
base=['rsync','-a','--partial','-e',transport]
receipt=source/'transfer-complete.json'
for attempt in range(3):
    try:
        subprocess.run(base+['--exclude=/transfer-complete.json',str(source)+'/', 'root@87.192.101.6:./'],check=True)
        receipt.write_text(json.dumps({'status':'complete','source_worker':'node-b','planned':26,'unix_s':time.time()},indent=2)+'\n')
        subprocess.run(base+[str(receipt),'root@87.192.101.6:./'],check=True)
        break
    except subprocess.CalledProcessError:
        if attempt==2:raise
        time.sleep(30)
