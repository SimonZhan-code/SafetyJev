"""Local inference endpoint for the pinned native visual SafetyJev checkpoint."""
import argparse
import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
import json
from pathlib import Path
import time


def main():
    import numpy as np
    from PIL import Image
    import torch
    from jev.visual_model import VisualDecisionModel
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--config',required=True)
    parser.add_argument('--calibration',required=True)
    parser.add_argument('--port',type=int,default=8792)
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    calibration=json.loads(Path(args.calibration).read_text())
    temperature=calibration['temperature'][config['calibration_step']]
    priors={q['id']:calibration['prior_log_odds'][q['id']]['log_odds'] for q in config['queries']}
    model=VisualDecisionModel.load(args.checkpoint,device='cuda:0')
    if model.model_config['revision']!=config['base_model_revision']:
        raise ValueError('Unexpected base revision')
    model.eval()
    questions=[{'id':q['id'],'question':q['question']} for q in config['queries']]

    @torch.inference_mode()
    def infer(images):
        observations={name:torch.from_numpy(np.array(img.convert('RGB'),copy=True)).permute(2,0,1)[None,None]
                      .expand(len(questions),1,3,img.height,img.width) for name,img in images.items()}
        torch.cuda.synchronize()
        started=time.perf_counter()
        logits=model([q['question'] for q in questions],observations)
        scores=logits[:,1]-logits[:,0]
        calibrated=scores/temperature+torch.tensor([priors[q['id']] for q in questions],device=scores.device)
        raw=torch.sigmoid(scores).cpu().tolist()
        calibrated=torch.sigmoid(calibrated).cpu().tolist()
        values=scores.cpu().tolist()
        torch.cuda.synchronize()
        return {'checkpoint_revision':config['checkpoint_revision'],
                'answers':{q['id']:{'raw_yes':raw[i],'calibrated_yes':calibrated[i],'logit':values[i]}
                           for i,q in enumerate(questions)},
                'server_latency_s':time.perf_counter()-started,'input_tokens':model.last_input_tokens}

    warm=infer({camera:Image.new('RGB',(256,256)) for camera in ('overview','wrist')})
    health={**config,'status':'ready','method':model.model_config['method'],
            'batch_size':len(questions),'calibration_temperature':temperature,
            'checkpoint_temperature':float(model.temperature),
            'warmup_latency_s':warm['server_latency_s']}
    print(json.dumps({'status':'ready','port':args.port,'warmup_latency_s':warm['server_latency_s']}),flush=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send_json(self,value,status=200):
            blob=json.dumps(value,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(blob)));self.end_headers();self.wfile.write(blob)
        def do_GET(self):
            self.send_json(health if self.path=='/health' else {'error':'not_found'},200 if self.path=='/health' else 404)
        def do_POST(self):
            if self.path!='/classify':self.send_json({'error':'not_found'},404);return
            try:
                length=int(self.headers['Content-Length'])
                if not 0<length<16*1024*1024:raise ValueError('Invalid body size')
                body=json.loads(self.rfile.read(length))
                if set(body)!={'images','questions'} or body['questions']!=questions:
                    raise ValueError('Expected exact trained question definitions')
                if set(body['images'])!={'overview','wrist'}:raise ValueError('Expected both cameras')
                images={}
                for camera,value in body['images'].items():
                    blob=base64.b64decode(value,validate=True)
                    if not blob.startswith(b'\x89PNG\r\n\x1a\n'):raise ValueError('Expected PNG')
                    image=Image.open(BytesIO(blob))
                    if image.width*image.height>1024*1024:raise ValueError('Image too large')
                    images[camera]=image.convert('RGB')
                result=infer(images)
                self.send_json(result)
            except Exception as exc:
                print('classification_error',type(exc).__name__,flush=True)
                self.send_json({'error':type(exc).__name__},500)
    HTTPServer(('127.0.0.1',args.port),Handler).serve_forever()


if __name__=='__main__':main()
