"""Stop each Vast instance using only its own credential after 200 base cases.

Node A archives the complete base results and permits a local-backup grace period.
Node B reads the non-secret ready record and acknowledges before either stops.
There is no recurring chat automation and no destruction of either instance.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/workspace/SafetyJev/artifacts/base-sweep-20261006')
DOMAIN=ROOT/'artifacts/domain-evaluation-20261006'
ART=ROOT/'artifacts'
IDS={'node-a':'54498592','node-b':'54533396'}
ARCHIVE=ART/'base-complete-20261006.tar.gz'
READY=ART/'base-stop-ready.json'


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def all_base_terminal(base_progress,jar_progress,totals):
    return (base_progress.get('status')=='finished' and jar_progress.get('status')=='finished'
            and base_progress.get('planned')==174 and jar_progress.get('planned')==26
            and totals.get('planned')==200 and totals.get('completed',0)+totals.get('failed',0)==200
            and totals.get('pending',0)==0 and totals.get('running',0)==0)


def own_stop(worker,status):
    ident=os.environ.get('CONTAINER_ID');token=os.environ.get('CONTAINER_API_KEY')
    if ident!=IDS[worker] or not token:raise RuntimeError('Instance identity or local credential check failed')
    # Never log the command or raw stderr: the credential must stay on this node.
    for attempt in range(1,31):
        save(status,{'status':'requesting_instance_stop','worker':worker,'instance_id':ident,'attempt':attempt})
        try:
            result=subprocess.run(['vastai','stop','instance',ident,'--api-key',token],capture_output=True,text=True,timeout=90)
        except subprocess.TimeoutExpired:
            time.sleep(30);continue
        if result.returncode==0 and f'stopping instance {ident}.' in result.stdout:
            save(status,{'status':'stop_request_accepted','worker':worker,'instance_id':ident,'unix_s':time.time()})
            return
        save(status,{'status':'stop_request_failed_retrying','worker':worker,'instance_id':ident,'attempt':attempt,'exit_code':result.returncode})
        time.sleep(30)
    raise RuntimeError('Vast stop request failed after 30 attempts; no credentials logged')


def peer_stopped_receipt(path,expected_worker="node-b"):
    """Validate the operator-saved, non-secret Vast status for the completed peer."""
    data=read(Path(path))
    absent=data.get('instance_found') is False
    stopped=(data.get('actual_status') in ('exited','stopped')
             and data.get('intended_status')=='stopped' and data.get('cur_state')=='stopped')
    if (data.get('source')!='vastai show instance' or str(data.get('id'))!=IDS[expected_worker]
            or not (absent or stopped)
            or not isinstance(data.get('checked_unix_s'),(int,float))
            or not 0<data['checked_unix_s']<=time.time()+60):
        raise RuntimeError('Completed peer is not confirmed stopped')
    return data


def node_a(status,backup_grace_s,peer_stopped=None,coordinator_worker="node-a"):
    peer_worker="node-b" if coordinator_worker=="node-a" else "node-a"
    peer=peer_stopped_receipt(peer_stopped,peer_worker) if peer_stopped else None
    if coordinator_worker!="node-a" and not peer:raise RuntimeError("Migrated coordinator needs inactive-peer evidence")
    spec=importlib.util.spec_from_file_location('report',ROOT/'scripts/reporting/build-id-ood-report.py')
    report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)
    if peer:save(ART/'completed-peer-stopped.json',peer)
    resources=read(ART/'domain-sweep-resources.json')
    save(status,{'status':'waiting_for_200_base_cases','worker':coordinator_worker})
    while True:
        a=read(BASE/'progress.json');b=read(DOMAIN/'jar-base/progress.json')
        if a.get('status')=='finished' and b.get('status')=='finished' and read(DOMAIN/'jar-base/transfer-complete.json').get('status')=='complete':
            data=report.collect(BASE,DOMAIN,resources,levels=['base'])
            if all_base_terminal(a,b,data['totals']):break
        time.sleep(20)
    # The original base process records finished just before its final service cleanup.
    while True:
        service=subprocess.run(['supervisorctl','status','safetyjev-base-sweep'],capture_output=True,text=True).stdout
        if 'RUNNING' not in service:break
        time.sleep(5)
    deadline=time.monotonic()+600
    while read(ART/'base-report-status.json').get('status') not in ['pdf_generated_pending_visual_review_and_local_backup','audit_review_required','failed']:
        if time.monotonic()>deadline:break
        time.sleep(10)
    save(status,{'status':'archiving','totals':data['totals'],'report_status':read(ART/'base-report-status.json')})
    inputs=[(BASE,'base-sweep-20261006'),(DOMAIN/'jar-base','domain-evaluation-20261006/jar-base'),
            (Path('/workspace/ManiGuard/outputs/eval_logs'),'maniguard-eval-logs'),
            (ART/'domain-sweep-resources.json','domain-sweep-resources.json'),
            (ROOT/'configs/two-node-assignments.json','two-node-assignments.json'),
            (ART/'base-report-status.json','base-report-status.json'),(ROOT/'output/pdf','output/pdf'),
            (BASE.parent/'base-migration-20261006.json','base-migration-20261006.json'),
            (ART/'completed-peer-stopped.json','completed-peer-stopped.json')]
    meta=read(ART/'base-archive-ready.json')
    reusable=False
    if ARCHIVE.exists() and meta.get('totals')==data['totals'] and meta.get('bytes')==ARCHIVE.stat().st_size:
        with ARCHIVE.open('rb') as stream:reusable=hashlib.file_digest(stream,'sha256').hexdigest()==meta.get('sha256')
    if not reusable:
        tmp=ARCHIVE.with_suffix('.partial')
        with tarfile.open(tmp,'w:gz',compresslevel=1) as tar:
            for path,name in inputs:
                if path.exists():tar.add(path,arcname=name)
        with tmp.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        tmp.replace(ARCHIVE)
        meta={'status':'archive_ready','archive':str(ARCHIVE),'sha256':digest,'bytes':ARCHIVE.stat().st_size,
              'totals':data['totals'],'unix_s':time.time(),'report_status':read(ART/'base-report-status.json')}
        save(ART/'base-archive-ready.json',meta)
    digest=meta['sha256']
    save(status,{**meta,'status':'waiting_for_local_backup','backup_grace_s':backup_grace_s})
    deadline=time.monotonic()+backup_grace_s
    while time.monotonic()<deadline:
        if read(ART/'local-base-backup-complete.json').get('sha256')==digest:break
        time.sleep(10)
    local=read(ART/'local-base-backup-complete.json').get('sha256')==digest
    # Keep the archive and disks even if the Mac is unavailable; never burn GPU time indefinitely.
    save(READY,{'status':'all_200_base_cases_saved','totals':data['totals'],'archive_sha256':digest,
                'local_backup_confirmed':local,'unix_s':time.time()})
    if peer:
        save(ART/'completed-peer-stopped.json',peer)
        save(status,{'status':'completed_peer_already_stopped','local_backup_confirmed':local,'peer_instance_id':peer['id']})
    else:
        save(status,{'status':'waiting_for_node_b_ack','local_backup_confirmed':local})
        while read(DOMAIN/'jar-base/node-b-stop-ack.json').get('archive_sha256')!=digest:time.sleep(10)
    own_stop(coordinator_worker,status)


def node_b(status):
    ssh='ssh -p 15019 -i /workspace/bootstrap-copy/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15'
    local=ART/'node-a-base-stop-ready.json'
    save(status,{'status':'waiting_for_all_200_base_cases','worker':'node-b'})
    while True:
        ready=read(local)
        if ready.get('status')=='all_200_base_cases_saved' and ready.get('totals',{}).get('planned')==200:break
        if read(DOMAIN/'jar-base/progress.json').get('status')=='finished' and read(DOMAIN/'jar-base/transfer-complete.json').get('status')=='complete':
            result=subprocess.run(['rsync','-a','-e',ssh,'root@87.192.101.6:SafetyJev-ood-20261006/artifacts/base-stop-ready.json',str(local)],capture_output=True,text=True)
            ready=read(local) if result.returncode==0 else {}
            if ready.get('status')=='all_200_base_cases_saved' and ready.get('totals',{}).get('planned')==200:break
        time.sleep(20)
    while True:
        state=subprocess.run(['supervisorctl','status','safetyjev-domain-queue'],capture_output=True,text=True).stdout
        if 'RUNNING' not in state:break
        time.sleep(5)
    ack=DOMAIN/'jar-base/node-b-stop-ack.json'
    save(ack,{'status':'completion_received_will_stop_locally','instance_id':IDS['node-b'],'archive_sha256':ready['archive_sha256']})
    transport='ssh -p 15019 -i /workspace/bootstrap-copy/id_result -o BatchMode=yes -o ConnectTimeout=15'
    # Once this receipt lands, either node may stop without a cross-host secret.
    delivered=ART/'node-b-stop-ack-delivered.json'
    if read(delivered).get('archive_sha256')!=ready['archive_sha256']:
        subprocess.run(['rsync','-a','-e',transport,str(ack),'root@87.192.101.6:./'],check=True,timeout=120)
        save(delivered,{'archive_sha256':ready['archive_sha256']})
    own_stop('node-b',status)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',choices=list(IDS),required=True)
    p.add_argument('--backup-grace-s',type=int,default=1200)
    p.add_argument('--peer-stopped-receipt',type=Path,help='Operator-verified provider status of the inactive peer')
    p.add_argument('--coordinator',action='store_true',help='Own both saved base roots and finish archive/stop locally after migration')
    args=p.parse_args()
    if args.peer_stopped_receipt and args.worker!='node-a' and not args.coordinator:p.error('Migrated peer receipt requires --coordinator')
    if os.environ.get('CONTAINER_ID')!=IDS[args.worker]:raise RuntimeError('Wrong instance for selected worker')
    if not os.environ.get('CONTAINER_API_KEY'):raise RuntimeError('Local instance credential unavailable')
    ART.mkdir(parents=True,exist_ok=True)
    status=ART/'automatic-stop-status.json'
    try:
        if args.worker=='node-a' or args.coordinator:node_a(status,args.backup_grace_s,args.peer_stopped_receipt,args.worker)
        else:node_b(status)
    except Exception as exc:
        # Do not render arbitrary exception text: subprocess arguments can contain a credential.
        save(status,{'status':'failed','worker':args.worker,'error_type':type(exc).__name__})
        raise SystemExit('Automatic stop failed; inspect the recorded stage (credentials omitted).')


if __name__=='__main__':main()
