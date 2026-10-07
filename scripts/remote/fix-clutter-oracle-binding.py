"""Pinned correction for exact scene-object names in ManiGuard LTL bindings."""
import argparse
import hashlib
import json
from pathlib import Path

ORIGINAL_SHA256='690e12205c2dc068c37e2acf25ec9e05821d7581a016929720062f9379e6e37e'
ANCHOR='''        base = prefix.split(".n.")[0]
'''
INSERT='''        # Exact scene-object names must remain exact resolver keys. Appending
        # an instance suffix makes Clutter predicates resolve an empty scope.
        if not any(char in pat for char in "*?["):
            exact = env.scene.object_registry("name", pat)
            if exact is not None:
                active[pat] = exact
                continue
        base = prefix.split(".n.")[0]
'''
END='''            active[f"{prefix}_{i}"] = obj
    return active
'''
REPLACEMENT='''            active[f"{prefix}_{i}"] = obj
    unresolved = [pat for pat in patterns
                  if not any(fnmatch.fnmatch(key, pat) for key in active)]
    if unresolved:
        raise ValueError("Unresolved LTL object patterns: " + repr(sorted(unresolved)))
    return active
'''


def corrected_source(source):
    if hashlib.sha256(source.encode()).hexdigest()!=ORIGINAL_SHA256:
        raise ValueError('Unexpected ManiGuard source revision; refuse unreviewed patch')
    if source.count(ANCHOR)!=1 or source.count(END)!=1:raise ValueError('Patch anchors differ')
    return source.replace(ANCHOR,INSERT).replace(END,REPLACEMENT)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    args=p.parse_args();old=args.source.read_text();new=corrected_source(old)
    backup=args.source.with_suffix('.before-clutter-binding.py')
    if backup.exists() and backup.read_text()!=old:raise ValueError('Conflicting source backup')
    backup.write_text(old);args.source.write_text(new)
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps({'correction':'Preserve exact object names; reject unresolved safety scopes',
        'source':str(args.source),'original_sha256':ORIGINAL_SHA256,
        'patched_sha256':hashlib.sha256(new.encode()).hexdigest(),
        'scope':'Clutter worker only; policy and benchmark propositions unchanged'},indent=2)+'\n')


if __name__=='__main__':main()
