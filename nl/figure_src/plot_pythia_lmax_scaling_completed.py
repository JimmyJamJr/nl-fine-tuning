#!/usr/bin/env python3
"""Saturating exponential fits for the fitted Pythia families (410M, 1.4B, 2.8B) on log-log axes.
Observed = solid; saturating exponential fit = dotted, over the observed range only (no extrapolation).
Full fitted equation reported in the legend; the ceiling L_max is the leading coefficient."""
import os
os.environ.setdefault("OMP_NUM_THREADS","1")
import sys, json, numpy as np
from math import log10, floor
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
# Drawn at printed size (0.8 text width) and included at 100% scale; no title (the caption says it).
sys.path.insert(0, "/home/huan2073/nl-fine-tuning/nl/figure_src")
import print_style as PS
PS.apply()
from scipy.optimize import curve_fit, differential_evolution
SCRATCH="/scratch/gautschi/huan2073"
OUT="/home/huan2073/nl-fine-tuning/nl/plots/paper_fig5"; os.makedirs(OUT, exist_ok=True)
MODELS=[
 ("Pythia-410M",405_334_016,"#85216b",["10696605","10730890","11114263","11426007","12494239"]),
 ("Pythia-1.4B",1_414_647_808,"#d84c3e",["10579766","10682439","10730893","11696279","15232695"]),  # 15232695: completed Aug 17 at L=76 (5M-PFLOP budget cap)
 ("Pythia-2.8B",2_775_208_960,"#fca50a",["10580483","10694584","10730892","11426089","11647896"]),
]
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
    pf=np.array(pf);Ls=np.array(Ls);m=(pf>1)&(Ls>=1);pf,Ls=pf[m],Ls[m];o=np.argsort(pf);return pf[o],Ls[o]
def f_wb(x,L,c0,k):return L*(1-np.exp(-np.power(np.maximum(x,0)/c0,k)))
WBB=([50,1e3,0.05],[1e6,1e12,3.0])
def fit(x,y):
    de=differential_evolution(lambda p:float(np.sum((f_wb(x,*p)-y)**2)) if np.all(np.isfinite(f_wb(x,*p))) else 1e30,
        [(50,1e6),(1e3,1e12),(0.05,3.0)],seed=42,maxiter=200,tol=1e-10,popsize=30,polish=True,init="sobol")
    po,_=curve_fit(f_wb,x,y,p0=list(de.x),bounds=([50,1e3,0.05],[1e6,1e12,3.0]),maxfev=40000);return po
def rmse(y,yh):return float(np.sqrt(np.mean((y-yh)**2)))
def r2(y,yh):return float(1-np.sum((y-yh)**2)/np.sum((y-y.mean())**2))
# METRIC=r2 reproduces the paper's Figure 10 legend (R^2 with 95% block-bootstrap CI);
# the default (rmse) is the RMSE variant. Output filename gets a "_r2" suffix in r2 mode.
METRIC=os.environ.get("METRIC","lmax").lower()   # default: name + ceiling only (paper legend); rmse|r2 add equation + metric with CI
metric=r2 if METRIC=="r2" else rmse
MLABEL="$R^2$" if METRIC=="r2" else "RMSE"
MFMT,CFMT=(".4f",".3f") if METRIC=="r2" else (".2f",".2f")
SUFFIX="_r2" if METRIC=="r2" else ""
def sci(v):
    e=int(floor(log10(v))); m=v/10**e; return f"{m:.1f}{{\\times}}10^{{{e}}}"
def mblock(n,rng):
    bl=max(2,int(round(n**(1/3.)))); nb=int(np.ceil(n/bl)); idx=[]
    for _ in range(nb): s=int(rng.integers(0,n-bl+1)); idx.extend(range(s,s+bl))
    return np.array(idx[:n])
def rmse_ci(x,y,p0,nb=400,seed=7):
    lo_,hi_=WBB; rng=np.random.default_rng(seed); v=[]
    for _ in range(nb):
        idx=mblock(len(x),rng); xb,yb=x[idx],y[idx]; best=None
        for s0 in [list(p0)]+[[rng.uniform(l,h) for l,h in zip(lo_,hi_)] for _ in range(4)]:
            try:
                po,_=curve_fit(f_wb,xb,yb,p0=s0,bounds=WBB,maxfev=15000); yh=f_wb(xb,*po)
                if not np.all(np.isfinite(yh)): continue
                rss=float(np.sum((yh-yb)**2))
                if best is None or rss<best[1]: best=(po,rss)
            except: continue
        if best: v.append(metric(yb,f_wb(xb,*best[0])))
    return np.percentile(v,2.5),np.percentile(v,97.5)

fig,ax=plt.subplots(figsize=(4.4,2.4))
for nm,N,col,dirs in MODELS:
    x,y=load(dirs,N); Lmax,C0,k=fit(x,y); yh=f_wb(x,Lmax,C0,k)
    lo,hi=(float("nan"),float("nan")) if METRIC in ("lmax","none") else rmse_ci(x,y,(Lmax,C0,k))  # CI only needed when displayed
    print(f"{nm}: FULL PARAMS Lmax={Lmax:.2f} C0={C0:.4g} k={k:.4f}  n={len(x)} L_end={y.max():.0f}")
    xs=np.logspace(np.log10(x.min()),np.log10(x.max()),300)
    eqn=f"$L\\approx{Lmax:.0f}\\,(1-e^{{-(C/{sci(C0)})^{{{k:.3f}}}}})$"
    # Legend modes: METRIC=lmax -> "name (L_max ~ N)" only, the paper's Fig. 10 style (matches Fig. 5:
    # no metrics, no formulas; full equations live in the appendix table). METRIC=none -> equation only.
    # METRIC=rmse|r2 -> equation + metric with 95% CI.
    if METRIC=="lmax":   lbl = f"{nm}  ($L_{{\\max}}\\approx{Lmax:.0f}$)"
    elif METRIC=="none": lbl = f"{nm}:  {eqn}"
    else:                lbl = f"{nm}:  {eqn},  {MLABEL}={metric(y,yh):{MFMT}} [{lo:{CFMT}}, {hi:{CFMT}}]"
    ax.plot(x,y,"-",color=col,lw=PS.LW,zorder=3,label=lbl)
    ax.plot(xs,f_wb(xs,Lmax,C0,k),":",color="#222222",lw=PS.LW_THIN,zorder=4)
    print(f"{nm}: Lmax={Lmax:.0f} k={k:.3f} {METRIC.upper()}={metric(y,yh):{MFMT}} CI[{lo:{CFMT}},{hi:{CFMT}}]")
ax.set_xscale("log"); ax.set_yscale("log")
PS.plain_log_ticks(ax)   # 10, 100, 1k, ... instead of 10^1 (superscripts print at 4.9pt)
ax.set_xlabel("Cumulative Compute (PFLOPs, log scale)")
ax.set_ylabel("Achieved Lookahead $L$ (log scale)")
ax.grid(True,which="both",alpha=0.25)
ax.legend(loc="upper left")
fig.tight_layout(pad=0.1)
fig.savefig(f"{OUT}/pythia_lmax_scaling{SUFFIX}_completed.png",dpi=200,bbox_inches="tight")
fig.savefig(f"{OUT}/pythia_lmax_scaling{SUFFIX}_completed.pdf",bbox_inches="tight")
print(f"Saved -> {OUT}/pythia_lmax_scaling{SUFFIX}.(png,pdf)")
