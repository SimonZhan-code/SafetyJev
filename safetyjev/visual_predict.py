"""Inference for a visual current-state Noul checkpoint and one camera pair."""
import argparse
import json

import numpy as np
from PIL import Image
import torch


def score_images(model, overview, wrist, questions):
    if not questions:raise ValueError("Provide named natural-language questions")
    observations={}
    for camera,image in (("overview",overview),("wrist",wrist)):
        pixels=torch.from_numpy(np.array(image.convert("RGB"),copy=True)).permute(2,0,1)
        observations[camera]=pixels[None,None].expand(len(questions),1,*pixels.shape)
    values=model.probabilities(list(questions.values()),observations).cpu().tolist()
    return {"answers":{name:{"type":"noul","noul":values[i][1]} for i,name in enumerate(questions)},
            "input_mode":"current_dual_camera","temperature":float(model.temperature)}


def main(argv=None):
    from jev.visual_model import VisualDecisionModel
    parser=argparse.ArgumentParser(description="Score current images without generating text")
    parser.add_argument("--checkpoint",required=True);parser.add_argument("--overview",required=True)
    parser.add_argument("--wrist",required=True);parser.add_argument("--question",required=True)
    parser.add_argument("--device",default="cuda:0")
    args=parser.parse_args(argv)
    model=VisualDecisionModel.load(args.checkpoint,device=args.device)
    with Image.open(args.overview) as overview,Image.open(args.wrist) as wrist:
        print(json.dumps(score_images(model,overview,wrist,{"predicate":args.question}),indent=2))


if __name__=="__main__":main()
