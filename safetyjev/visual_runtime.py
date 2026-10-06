"""Current-camera predicate classification beside an unchanged ManiGuard policy.

Run with ``python -m safetyjev.visual_runtime --help``. This is intentionally
separate from future-action-window forecasting and guard intervention.
"""
import argparse
import base64
import copy
import hashlib
import json
import math
from pathlib import Path
import time
from urllib.request import Request, urlopen

from .capture import ShadowEpisode
from .io import append_jsonl, read_jsonl, write_json
from .metrics import binary_metrics, quantile


def predicate_label(query, ap):
    conditions = query.get('all_of', {query.get('ap'): query.get('yes_if')})
    for name, expected in conditions.items():
        if type(expected) is not bool or type(ap.get(name)) is not bool:
            raise ValueError('Missing or invalid current AP: ' + str(name))
    value = all(ap[name] == expected for name, expected in conditions.items())
    return int(value if 'all_of' not in query or query.get('yes_if', True) else not value)


def image_request(frames, queries):
    """Build from image bytes and questions only; no simulator state is accepted."""
    return {'images': {name: base64.b64encode(blob).decode() for name, blob in frames.items()},
            'questions': [{'id': q['id'], 'question': q['question']} for q in queries]}


def validate_answers(response, queries):
    answers = response['answers']
    if set(answers) != {q['id'] for q in queries}:
        raise ValueError('Missing or unexpected predicate predictions')
    for answer in answers.values():
        for field in ('raw_yes', 'calibrated_yes', 'logit'):
            value = answer[field]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('Invalid classifier score')
            if field != 'logit' and not 0 <= value <= 1:
                raise ValueError('Classifier probability outside [0,1]')
    return answers


class VisualClassificationEpisode(ShadowEpisode):
    execution_mode = 'visual_classification_shadow'

    def __init__(self, *args):
        from .capture import OPTIONS
        self.visual_config = OPTIONS['visual_config']
        self.queries = self.visual_config['queries']
        self.classification_count = self.classification_failures = 0
        self.ready = False
        super().__init__(*args)
        self.provenance.update(label_semantics='Current AP truth at the image step; not cumulative LTL rejection or future-chunk risk',
                               visual_classifier=self.visual_config)
        write_json(self.directory/'episode.json', self.provenance)
        self.ready = True
        self.classify(0, args[3])

    def before_action(self, *args):
        pass  # No action modification and no future-window predictions.

    def after_step(self, step, monitor, observation):
        super().after_step(step, monitor, observation)
        self.current_ap = copy.deepcopy(monitor._ltl_log[-1]['ap']) if not self.invalid else None
        if self.ready and step % self.visual_config['sample_stride'] == 0:
            self.classify(step, observation)

    def classify(self, step, observation):
        if self.current_ap is None:
            raise ValueError('Cannot classify with invalid aligned monitor state')
        frames, images = {}, {}
        for camera, key in [('overview','overview_image'), ('wrist','wrist_images')]:
            relative = f'frames/{step:07d}-{camera}.png'
            self.imageio.imwrite(self.directory/relative, observation[key])
            frames[camera] = (self.directory/relative).read_bytes()
            images[camera] = {'path': relative, 'sha256': hashlib.sha256(frames[camera]).hexdigest()}
        sample_id = f'{self.episode_id}:{step}'
        # Evaluator labels never enter image_request or the scorer process.
        append_jsonl(self.directory/'classification-labels.jsonl', {
            'sample_id': sample_id, 'step': step, 'labels': {q['id']: predicate_label(q,self.current_ap) for q in self.queries}})
        body = json.dumps(image_request(frames,self.queries)).encode()
        append_jsonl(self.directory/'classification-inputs.jsonl', {
            'sample_id': sample_id, 'step': step, 'images': images,
            'questions': [{'id': q['id'], 'question': q['question']} for q in self.queries],
            'request_sha256': hashlib.sha256(body).hexdigest()})
        started = time.perf_counter()
        record = {'sample_id': sample_id, 'step': step}
        try:
            request = Request(self.visual_config['endpoint']+'/classify',data=body,headers={'Content-Type':'application/json'})
            with urlopen(request,timeout=self.visual_config.get('timeout',120)) as response:
                data = json.load(response)
            if data['checkpoint_revision'] != self.visual_config['checkpoint_revision']:
                raise ValueError('Classifier checkpoint revision mismatch')
            record.update(answers=validate_answers(data,self.queries), error=None,
                          server_latency_s=data['server_latency_s'], input_tokens=data['input_tokens'])
        except Exception as exc:
            self.classification_failures += 1
            record.update(answers={}, error=type(exc).__name__+': '+str(exc))
        record['latency_s'] = time.perf_counter()-started
        self.classification_count += 1
        append_jsonl(self.directory/'classification-predictions.jsonl',record)

    def finish(self, result):
        result['safetyjev_classification'] = {'samples':self.classification_count,'failed_samples':self.classification_failures}
        super().finish(result)


def report(root, output):
    paths = sorted(Path(root).glob('*/episode.json'))
    if not paths:
        raise ValueError('No classification episodes')
    pairs = {'raw':{}, 'calibrated':{}}
    episodes, latencies, signatures, incomplete = [], [], set(), []
    expected = failures = 0
    for path in paths:
        ep=path.parent
        meta=json.loads(path.read_text())
        if not (ep/'complete.json').exists():
            incomplete.append(ep.name); continue
        completion=json.loads((ep/'complete.json').read_text())
        if completion.get('status')!='completed' or completion.get('monitor_valid') is not True:
            raise ValueError('Invalid completed classification episode: '+ep.name)
        signatures.add(json.dumps(meta['visual_classifier'],sort_keys=True))
        labels=read_jsonl(ep/'classification-labels.jsonl')
        predictions=read_jsonl(ep/'classification-predictions.jsonl')
        if len({r['sample_id'] for r in labels})!=len(labels):
            raise ValueError('Duplicate classification label ID')
        lookup={r['sample_id']:r for r in predictions}
        if len(lookup)!=len(predictions) or set(lookup)-{r['sample_id'] for r in labels}:
            raise ValueError('Duplicate or unmatched classifier predictions')
        for label in labels:
            expected += len(label['labels'])
            pred=lookup.get(label['sample_id'])
            if not pred or pred.get('error'):
                failures += len(label['labels']); continue
            validate_answers(pred,meta['visual_classifier']['queries'])
            latencies.append(pred['latency_s'])
            if set(label['labels'])!={q['id'] for q in meta['visual_classifier']['queries']}:
                raise ValueError('Missing or unexpected classification label')
            for q,y in label['labels'].items():
                if type(y) is not int or y not in (0,1):raise ValueError('Expected binary predicate label')
                for mode, field in [('raw','raw_yes'),('calibrated','calibrated_yes')]:
                    pairs[mode].setdefault(q,[]).append((y,pred['answers'][q][field]))
        episodes.append({'episode_id':ep.name,'scene':meta['scene_name'],
                         'samples':len(labels),'outcome':json.loads((ep/'maniguard_result.json').read_text())})
    if len(signatures)>1:raise ValueError('Mixed classifier configurations')
    result={'scope':'Current-state predicate classification; Yes polarity depends on question',
            'threshold':.5,'episodes':episodes,'incomplete_episode_ids':incomplete,
            'expected_classifications':expected,'failed_or_missing_classifications':failures,
            'coverage':(expected-failures)/expected if expected else None,
            'frame_latency_s':{'n':len(latencies),'p50':quantile(latencies,.5),'p95':quantile(latencies,.95)},
            'classifier':json.loads(next(iter(signatures))) if signatures else None}
    for mode, by_query in pairs.items():
        result[mode]={'by_question':{q:binary_metrics(v,.5) for q,v in by_query.items()},
                      'micro':binary_metrics([pair for values in by_query.values() for pair in values],.5)}
    write_json(output,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    run=sub.add_parser('capture')
    run.add_argument('--maniguard-root',required=True);run.add_argument('--output',required=True)
    run.add_argument('--provenance',required=True);run.add_argument('--classifier-config',required=True)
    run.add_argument('benchmark_args',nargs=argparse.REMAINDER)
    rep=sub.add_parser('report');rep.add_argument('--episodes',required=True);rep.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.command=='report':
        result=report(args.episodes,args.output)
        print(json.dumps({k:result[k] for k in ('coverage','expected_classifications','frame_latency_s')},indent=2));return
    from .maniguard import launch, SOURCE_HASHES, COMMIT
    config=json.loads(Path(args.classifier_config).read_text())
    if type(config['sample_stride']) is not int or config['sample_stride']<1:raise ValueError('Invalid sample stride')
    with urlopen(config['endpoint']+'/health',timeout=10) as response: health=json.load(response)
    for field in ('checkpoint_revision','checkpoint_subdirectory','base_model_revision','loader_revision'):
        if health[field]!=config[field]:raise ValueError('Serving identity mismatch: '+field)
    if health['queries']!=config['queries']:raise ValueError('Serving question definitions mismatch')
    provenance=json.loads(Path(args.provenance).read_text())
    provenance.update(source_hashes=SOURCE_HASHES,maniguard_reference_commit=COMMIT)
    bench=args.benchmark_args[1:] if args.benchmark_args[:1]==['--'] else args.benchmark_args
    launch(args.maniguard_root,bench,{'output':args.output,'provenance':provenance,
           'execution_mode':'visual_classification','recheck_every':0,'visual_config':config})


if __name__=='__main__':main()
