"""Tiny on-disk cache for parsed training chains (compute vs lookahead arrays).
Key = tag + loader args + (path, size, mtime) of every loss_history.jsonl in the chain, so a run that
has grown (resumed/preempted job wrote more lines) is re-parsed automatically; everything else loads in ms.
Usage:  load = cache_wrap(load, "reinit")   # load(dirs, *args) -> tuple of numpy arrays
Set CHAIN_CACHE=0 to bypass."""
import os, json, hashlib, numpy as np
SCRATCH="/scratch/gautschi/huan2073"; CACHE_DIR=f"{SCRATCH}/audit_tmp/cache"
def _stats(dirs):
    out=[]
    for d in dirs:
        p=d if str(d).startswith("/") else f"{SCRATCH}/nl_output/search/job_{d}"
        f=os.path.join(p,"loss_history.jsonl")
        try: st=os.stat(f); out.append((f,st.st_size,int(st.st_mtime)))
        except FileNotFoundError: out.append((f,-1,-1))
    return out
def cache_wrap(fn, tag):
    if os.environ.get("CHAIN_CACHE","1")=="0": return fn
    os.makedirs(CACHE_DIR, exist_ok=True)
    def wrapped(dirs, *args):
        key=hashlib.sha1(json.dumps([tag,list(map(str,dirs)),[repr(a) for a in args],_stats(dirs)]).encode()).hexdigest()
        path=f"{CACHE_DIR}/{tag}_{key}.npz"
        if os.path.exists(path):
            z=np.load(path, allow_pickle=False); return tuple(z[f"a{i}"] for i in range(int(z["n"])))
        res=fn(dirs,*args); res=tuple(res) if isinstance(res,(tuple,list)) else (res,)
        tmp=path+".tmp.npz"; np.savez(tmp, n=len(res), **{f"a{i}":np.asarray(r) for i,r in enumerate(res)}); os.replace(tmp,path)
        return res
    return wrapped
