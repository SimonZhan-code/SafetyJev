"""One completion-dependent PDF build; not a recurring chat monitor."""
import importlib.util
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/workspace/SafetyJev/artifacts/base-sweep-20261006')
DOMAIN=ROOT/'artifacts/domain-evaluation-20261006'
OUT=ROOT/'output/pdf/maniguard-base-evaluation.pdf'
STATUS=ROOT/'artifacts/base-report-status.json'


def save(data):
    p=STATUS.with_suffix('.tmp');p.write_text(json.dumps(data,indent=2)+'\n');p.replace(STATUS)


def main():
    save({'status':'waiting_for_200_base_cases','hourly_chat_monitor':'disabled'})
    while True:
        a=BASE/'progress.json';b=DOMAIN/'jar-base/transfer-complete.json'
        if a.exists() and b.exists() and json.loads(a.read_text()).get('status')=='finished' and json.loads(b.read_text()).get('status')=='complete':break
        time.sleep(30)
    spec=importlib.util.spec_from_file_location('report',ROOT/'scripts/reporting/build-id-ood-report.py')
    report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)
    resources=json.loads((ROOT/'artifacts/domain-sweep-resources.json').read_text())
    data=report.collect(BASE,DOMAIN,resources,levels=['base'])
    if data['totals']['planned']!=200 or not data['complete']:raise RuntimeError('Base coverage is incomplete')
    issues=[{'family':r['family'],'warnings':r['warnings'],'audit':r['audit']} for r in data['rows'] if r['family']!='clutter' and (r['warnings'] or not r['classification'] or not r['audit'] or not r['audit'].get('passed'))]
    if issues:
        save({'status':'audit_review_required','issues':issues,'totals':data['totals']});return
    save({'status':'building','totals':data['totals']})
    report.render(data,OUT)
    OUT.with_suffix('.json').write_text(json.dumps(data,indent=2)+'\n')
    from pypdf import PdfReader
    reader=PdfReader(OUT)
    if not reader.pages or '200 planned cases' not in reader.pages[0].extract_text():raise RuntimeError('PDF content validation failed')
    save({'status':'pdf_generated_pending_visual_review_and_local_backup','pdf':str(OUT),'pages':len(reader.pages),'totals':data['totals'],'hourly_chat_monitor':'disabled'})


if __name__=='__main__':
    try:main()
    except Exception as exc:
        save({'status':'failed','error':type(exc).__name__+': '+str(exc)})
        raise
