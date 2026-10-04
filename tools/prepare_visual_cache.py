"""Prepare a disposable lossless image cache next to a downloaded data package."""
import argparse
import json
from pathlib import Path
from safetyjev.visual_cache import build_visual_cache,validate_cache

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--workers',type=int,default=1,help='Concurrent video decoders or verification processes (default: 1)')
    p.add_argument('--max-gib',type=float);p.add_argument('--verify-only',action='store_true')
    args=p.parse_args()
    result=validate_cache(args.output,args.package,workers=args.workers) if args.verify_only else build_visual_cache(
        args.package,args.output,workers=args.workers,max_bytes=None if args.max_gib is None else int(args.max_gib*2**30))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
