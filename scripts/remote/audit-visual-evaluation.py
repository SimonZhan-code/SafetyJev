"""Audit saved current-frame classification captures without loading GPU models."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from safetyjev.visual_runtime import image_request, predicate_label, validate_answers


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit(root, calibration_path):
    calibration = json.loads(calibration_path.read_text())
    episodes = []
    for ep in sorted((root / 'episodes').iterdir()):
        if not ep.is_dir():
            continue
        meta = json.loads((ep / 'episode.json').read_text())
        complete = json.loads((ep / 'complete.json').read_text())
        assert complete['status'] == 'completed' and complete['monitor_valid'] is True
        config = meta['visual_classifier']
        queries = meta.get('classification_queries', config['queries'])
        oracle = rows(ep / 'oracle.jsonl')
        assert [r['step'] for r in oracle] == list(range(complete['final_step'] + 1))
        assert all(r['valid'] is True for r in oracle)
        labels = rows(ep / 'classification-labels.jsonl')
        inputs = rows(ep / 'classification-inputs.jsonl')
        predictions = rows(ep / 'classification-predictions.jsonl')
        expected_steps = list(range(0, complete['final_step'] + 1, config['sample_stride']))
        assert [r['step'] for r in labels] == expected_steps
        assert len(labels) == len(inputs) == len(predictions)
        counts = {q['id']: 0 for q in queries}
        temperature = calibration['temperature'][config['calibration_step']]
        for label, request, prediction in zip(labels, inputs, predictions):
            step = label['step']
            assert request['step'] == prediction['step'] == step
            assert label['sample_id'] == request['sample_id'] == prediction['sample_id'] == f'{ep.name}:{step}'
            assert prediction['error'] is None
            validate_answers(prediction, queries)
            expected = {q['id']: predicate_label(q, oracle[step]['ap']) for q in queries}
            assert label['labels'] == expected
            frames = {}
            for camera, image in request['images'].items():
                frames[camera] = (ep / image['path']).read_bytes()
                assert hashlib.sha256(frames[camera]).hexdigest() == image['sha256']
            body = image_request(frames, queries)
            assert request['questions'] == body['questions']
            assert hashlib.sha256(json.dumps(body).encode()).hexdigest() == request['request_sha256']
            for q, y in expected.items():
                counts[q] += y
                answer = prediction['answers'][q]
                raw = 1 / (1 + math.exp(-answer['logit']))
                score = answer['logit'] / temperature + calibration['prior_log_odds'][q]['log_odds']
                calibrated = 1 / (1 + math.exp(-score))
                assert math.isclose(raw, answer['raw_yes'], rel_tol=1e-5, abs_tol=1e-7)
                assert math.isclose(calibrated, answer['calibrated_yes'], rel_tol=1e-5, abs_tol=1e-7)
        episodes.append({'episode_id': ep.name, 'scene': meta['scene_name'],
                         'valid_oracle_rows': len(oracle), 'sample_steps': expected_steps,
                         'frame_pairs': len(labels), 'yes_counts': counts})
    assert episodes, 'No captured episodes'
    return {'passed': True, 'episodes': episodes,
            'checks': ['contiguous valid monitor steps', 'stride and aligned sample IDs',
                       'current AP labels and question polarity', 'image SHA-256',
                       'exact image/question-only request SHA-256', 'complete predictions',
                       'raw sigmoid and frozen release calibration'],
            'limitation': 'Trace consistency does not independently validate physical AP truth or benchmark equivalence.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, args.calibration)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': result['passed'], 'episodes': len(result['episodes']),
                      'frame_pairs': sum(e['frame_pairs'] for e in result['episodes'])}))
