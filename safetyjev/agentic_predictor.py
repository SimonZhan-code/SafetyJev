"""Local native-predictor client. Only complete binary answer sets are valid."""
import hashlib
import json
from urllib.request import Request, urlopen
from .agentic_config import IDENTITY


def request_digest(questions, inputs):
    return hashlib.sha256(json.dumps({'questions': questions, 'inputs': inputs},
                                    sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_identity(health):
    if any(health.get(k) != v for k, v in IDENTITY.items()) or health.get('status') != 'ready':
        raise ValueError('Native predictor identity mismatch or service not ready')


class NativePredictor:
    def __init__(self, config):
        self.endpoint = config['endpoint'].rstrip('/')
        self.timeout = config['timeout']
        self.optimizations = {k: config.get(k, v) for k, v in
                              (('attention_backend', 'torch'), ('vision_cache', False))}

    def validate_health(self, health):
        validate_identity(health)
        observed = health.get('optimizations', {'attention_backend': 'torch', 'vision_cache': False})
        if any(observed.get(k) != v for k, v in self.optimizations.items()):
            raise ValueError('Native predictor optimization mismatch')

    def health(self):
        with urlopen(self.endpoint+'/health', timeout=self.timeout) as response:
            health = json.load(response)
        self.validate_health(health)
        return health

    def score(self, request_id, questions, inputs):
        body = {'request_id': request_id, 'questions': questions, 'inputs': inputs}
        request = Request(self.endpoint+'/predict', data=json.dumps(body, allow_nan=False).encode(),
                          headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        self.validate_health(result)
        if result.get('request_id') != request_id:
            raise ValueError('Mismatched native prediction ID')
        answers = result.get('answers')
        if not isinstance(answers, dict) or set(answers) != {q['id'] for q in questions}:
            raise ValueError('Incomplete native prediction')
        if any(not isinstance(a, dict) or type(a.get('violated')) is not bool for a in answers.values()):
            raise ValueError('Expected native Yes/No answers')
        # Probability magnitudes and server-provided free text never reach the loop.
        return {qid: value['violated'] for qid, value in answers.items()}
