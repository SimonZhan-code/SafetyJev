"""Describe all-No runs in a saved trace; never infer counterfactual speed/success."""
import argparse
import itertools
import json
from pathlib import Path
import tarfile


def analyze(rows):
    judgments={r['candidate_id']:r for r in rows if r['event']=='judgment' and r.get('valid')}
    proposals={}
    for r in rows:
        if r['event']=='proposal' and r.get('valid'):
            proposals.setdefault(r['decision_id'],[]).append(r)
    choices=[r for r in rows if r['event']=='selection']
    clear=[]; initial=[]
    for row in choices:
        candidates=proposals[row['decision_id']]
        def all_no(candidate):
            result=judgments.get(candidate['candidate_id'])
            return bool(result and result['answers'] and all(a=='No' for a in result['answers'].values()))
        clear.append(any(all_no(c) for c in candidates));initial.append(all_no(candidates[0]))
    runs=[len(list(group)) for value,group in itertools.groupby(clear) if value]
    return {'decisions':len(choices), 'initial_candidate_all_No':sum(initial),
        'any_scored_candidate_all_No':sum(clear),
        'selected_predicted_unsafe':sum(r['predicted_unsafe_selected'] for r in choices),
        'all_No_run_lengths':runs, 'longest_all_No_run':max(runs,default=0),
        'first_no_all_clear_step':next((r['start_step'] for r,v in zip(choices,clear) if not v),None),
        'scope':'Descriptive replay of observed old-condition candidates only. New local retries, '
                'skill instructions and strict stopping change trajectories. No counterfactual runtime, '
                'safety or success claim; predictor warning is not an oracle label.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive',type=Path);p.add_argument('--member',required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    import zstandard
    found=None
    with args.archive.open('rb') as f,zstandard.ZstdDecompressor().stream_reader(f) as z,tarfile.open(fileobj=z,mode='r|') as t:
        for member in t:
            if member.name==args.member and member.isfile():
                found=analyze([json.loads(line) for line in t.extractfile(member)])
                break
    if found is None:raise SystemExit('Requested event member not found')
    found['archive_member']=args.member
    args.output.write_text(json.dumps(found,indent=2)+'\n')


if __name__=='__main__':main()
