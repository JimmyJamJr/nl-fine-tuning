#!/usr/bin/env python3
"""Paper Figure 5 ALTERNATIVE layout: panel-per-MODEL (Qwen3-0.6B | Pythia-1.4B), each
overlaying the observed trajectory with BOTH fits (power-law dashed, saturating exponential dotted) on
one set of log-log axes. The residual strips that used to sit under each panel were removed
on 2026-09-22; the misfit is described in the text and in the goodness-of-fit table instead."""
import os
os.environ.setdefault("OMP_NUM_THREADS","1")
import sys, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Drawn at printed size (two panels at 0.8 text width, print_style.FULL2_80) and included at 100%
# scale. Each panel keeps a one-line model label; the two-line titles restated the caption.
sys.path.insert(0, "/home/huan2073/nl-fine-tuning/nl/figure_src")
import print_style as PS
PS.apply()
from scipy.optimize import curve_fit, differential_evolution
SCRATCH="/scratch/gautschi/huan2073"
OUT="/home/huan2073/nl-fine-tuning/nl/plots/paper_fig5"; os.makedirs(OUT, exist_ok=True)
MODELS=[("Qwen3-0.6B",596_049_408,["10696449","10730891","11426006"]),
        ("Pythia-1.4B",1_414_647_808,["10579766","10682439","10730893","11696279","15232695"])]
C_OBS="#333333"; C_PL="#d62728"; C_WB="#1f77b4"   # observed / power-law / saturating exponential
def load(dirs,N):
    f=6*N/1e15;seen=-1;cum=0;pf=[];Ls=[];prev=None;prevL=None
    for d in dirs:
        p=f"{SCRATCH}/nl_output/search/job_{d}/loss_history.jsonl"
        if not os.path.exists(p):continue
        for line in open(p):
            try:e=json.loads(line)
            except:continue
            s=e["step"]
            if s<=seen:continue
            seen=s;cum+=e.get("tokens",0);st=e.get("stage");L=e.get("effective_L")
            if st is None or L is None:continue
            if prev is not None and st!=prev:pf.append(cum*f);Ls.append(prevL)   # COMPLETED stage = previous entry's L
            prev=st;prevL=L
    pf=np.array(pf);Ls=np.array(Ls);m=(pf>1)&(Ls>=1);pf,Ls=pf[m],Ls[m];o=np.argsort(pf);pf,Ls=pf[o],Ls[o]
    # De-duplicate stage entries replayed by a training resume (Qwen: L=161,162 appear twice after the
    # rollback at step 79,991). Keep the first entry per L so n matches Appendix E (Qwen n=244).
    keep=[];seenL=set()
    for i,L in enumerate(Ls):
        if L not in seenL: seenL.add(L);keep.append(i)
    keep=np.array(keep);return pf[keep],Ls[keep]
def f_pl(x,a,b,c): return a*np.power(np.maximum(x,1e-3),b)+c
def f_wb(x,L,c0,k): return L*(1-np.exp(-np.power(np.maximum(x,0)/c0,k)))
def fit(func,lo,hi,x,y):
    de=differential_evolution(lambda p:float(np.sum((func(x,*p)-y)**2)) if np.all(np.isfinite(func(x,*p))) else 1e30,
        list(zip(lo,hi)),seed=42,maxiter=200,tol=1e-10,popsize=30,polish=True,init="sobol")
    try:
        po,_=curve_fit(func,x,y,p0=list(de.x),bounds=(lo,hi),maxfev=40000)
        if np.all(np.isfinite(func(x,*po))): return po
    except Exception: pass
    return np.array(de.x)
def r2(y,yh): return 1-np.sum((y-yh)**2)/np.sum((y-y.mean())**2)

def mblock_idx(n,rng):
    bl=max(2,int(round(n**(1/3.0)))); nb=int(np.ceil(n/bl)); idx=[]
    for _ in range(nb):
        s=int(rng.integers(0,n-bl+1)); idx.extend(range(s,s+bl))
    return np.array(idx[:n])
def ms_refit(func,lo,hi,x,y,seed_popt,rng,n_rand=5):
    best=None
    for p0 in [list(seed_popt)]+[[rng.uniform(l,h) for l,h in zip(lo,hi)] for _ in range(n_rand)]:
        try:
            po,_=curve_fit(func,x,y,p0=p0,bounds=(lo,hi),maxfev=20000); yh=func(x,*po)
            if not np.all(np.isfinite(yh)): continue
            rss=float(np.sum((yh-y)**2))
            if best is None or rss<best[1]: best=(po,rss)
        except Exception: continue
    return best[0] if best else np.array(seed_popt)
def r2_block_ci(func,lo,hi,x,y,popt0,n_boot=500,seed=7):
    rng=np.random.default_rng(seed); vals=[]
    for _ in range(n_boot):
        idx=mblock_idx(len(x),rng); xb,yb=x[idx],y[idx]
        po=ms_refit(func,lo,hi,xb,yb,popt0,rng); vals.append(r2(yb,func(xb,*po)))
    return np.percentile(vals,2.5), np.percentile(vals,97.5)

fig,axs=plt.subplots(1,2,figsize=PS.FULL2_80)
YLIM=(1.8,300)   # shared y-range across both panels (covers L=2..245)
for j,(nm,N,dirs) in enumerate(MODELS):
    x,y=load(dirs,N)
    pl=fit(f_pl,[1e-3,0.05,0.0],[100,1.5,50],x,y); wb=fit(f_wb,[50,1e3,0.05],[1e6,1e12,3.0],x,y)
    xs=np.logspace(np.log10(x.min()),np.log10(x.max()),400)
    ax=axs[j]
    print(f"{nm}: PL R2={r2(y,f_pl(x,*pl)):.4f} | SatExp R2={r2(y,f_wb(x,*wb)):.4f}")
    ax.plot(x,y,"-",color=C_OBS,lw=PS.LW,label="Observed")
    ax.plot(xs,f_pl(xs,*pl),"--",color=C_PL,lw=PS.LW,label="Power-law fit")
    ax.plot(xs,f_wb(xs,*wb),":",color=C_WB,lw=PS.LW+0.3,label="Saturating exponential fit")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(*YLIM)
    PS.plain_log_ticks(ax)   # 100, 1k, ... instead of 10^2 (superscripts print at 4.9pt)
    ax.set_title(nm)
    ax.grid(True,which="both",alpha=0.25)
    ax.set_xlabel("Cumulative Compute (PFLOPs)")
    if j==0: ax.set_ylabel("Achieved Lookahead $L$")   # same y-range in both panels: label once
    # Legend inside the RIGHT panel only: both panels carry the same three curves, so one
    # copy is enough. At print size the legend is wide enough to reach the Qwen curve near
    # 1e4-1e5 PFLOPs, while Pythia-1.4B stays below L~60, leaving its upper left empty.
    if j==1:
        ax.legend(loc="upper left")
fig.tight_layout(pad=0.1,w_pad=1.0)
fig.savefig(f"{OUT}/fig5_overlay_per_model_completed.png",dpi=200,bbox_inches="tight")
fig.savefig(f"{OUT}/fig5_overlay_per_model_completed.pdf",bbox_inches="tight")
print(f"Saved -> {OUT}/fig5_overlay_per_model.(png,pdf)")
