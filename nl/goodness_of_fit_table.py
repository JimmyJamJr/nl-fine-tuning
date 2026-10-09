#!/usr/bin/env python3
"""Comprehensive goodness-of-fit table mirroring the memorization-capacity report's
methodology, for all 6 runs (4 pretrained + 2 reinit) x all 7 families.

Per family per run we report:
  - fitted equation
  - n, k, dof (n-k)
  - SS_res (RSS), RMSE, max|res|
  - R^2(all), R^2(L>10)
  - AICc, BIC, dAICc (vs best family in that run), AICc weight
  - LOOCV-RMSE (leave-one-out cross-validated prediction error)
  - Durbin-Watson, runs-test p, Shapiro-Wilk p  (residual-pattern checks)

NOT computed: known-error chi-square goodness-of-fit and chi-square p. That test
needs an independent noise estimate (SEM from seed replicates); our curriculum
chains are single-seed, so no SEM is available. (The report relies on 3 seeds/point.)
"""
import json, os, numpy as np
from scipy.optimize import curve_fit, differential_evolution
from scipy import stats
SCRATCH = "/scratch/gautschi/huan2073"

RUNS = {
    "Qwen 0.6B":          ("qwen06b",   596_049_408,  ["10696449","10730891","11426006"]),
    "Pythia 410M":        ("pythia410m", 405_334_016, ["10696605","10730890","11114263","11426007","12494239"]),
    "Pythia 1.4B":        ("pythia14b", 1_414_647_808,["10579766","10682439","10730893","11696279",
            "15232695"]),
    "Pythia 2.8B":        ("pythia28b", 2_775_208_960,["10580483","10694584","10730892","11426089","11647896"]),
    "Pythia 1.4B reinit": ("pythia14b", 1_414_647_808,["local_20260501_125955_pythia14b_step1_REINIT_lr1e4_eff768","11427690","11649285"]),
    "Qwen 0.6B reinit":   ("qwen06b",   596_049_408,  ["11427737","11649286"]),
}
def load(dirs, n_params):
    factor=6*n_params/1e15; seen=-1; cum=0; pf=[]; Ls=[]; prev=None
    for d in dirs:
        p=f"{SCRATCH}/nl_output/search/job_{d}/loss_history.jsonl"
        if not os.path.exists(p): continue
        for line in open(p):
            try: e=json.loads(line)
            except: continue
            s=e["step"]
            if s<=seen: continue
            seen=s; cum+=e.get("tokens",0); st=e.get("stage"); L=e.get("effective_L")
            if st is None or L is None: continue
            if prev is not None and st!=prev: pf.append(cum*factor); Ls.append(L)
            if prev is None: prev=st
            elif st!=prev: prev=st
    pf=np.array(pf); Ls=np.array(Ls); m=(pf>1)&(Ls>=2)
    return pf[m], Ls[m]

def f_powerlaw(x,a,b,c): return a*np.power(np.maximum(x,1e-3),b)+c
def f_shiftedpl(x,a,d,b): return a*np.power(np.maximum(x+d,1e-6),b)
def f_satexp(x,Lmax,x0,k): return Lmax*(1-np.exp(-np.power(np.maximum(x,0)/x0,k)))
def f_satexp_voff(x,Lmax,x0,k,L0): return L0+Lmax*(1-np.exp(-np.power(np.maximum(x,0)/x0,k)))
def f_satexp_hoff(x,Lmax,x0,k,c): return Lmax*(1-np.exp(-np.power(np.maximum(x+c,0)/x0,k)))
def f_satpower(x,Lmax,a,b,c): return Lmax*(1-a*np.power(np.maximum(x+b,1e-6),c))
def f_log(x,a,c,b): return a*np.log(np.maximum(x+c,1e-9))+b

def eqn(slug,p):
    if slug=="power_law": a,b,c=p; return f"{a:.3g}·C^{b:.3f}" + (f" + {c:.2g}" if c>0.01 else "")
    if slug=="shifted_pl": a,d,b=p; return f"{a:.3g}·(C+{d:.2g})^{b:.3f}"
    if slug=="sat_exp": L,C0,k=p; return f"{L:.3g}·(1−exp(−(C/{C0:.2g})^{k:.3f}))"
    if slug=="sat_exp_voff": L,C0,k,L0=p; return f"{L0:.3g} + {L:.3g}·(1−exp(−(C/{C0:.2g})^{k:.3f}))"
    if slug=="sat_exp_hoff": L,C0,k,c=p; return f"{L:.3g}·(1−exp(−((C+{c:.2g})/{C0:.2g})^{k:.3f}))"
    if slug=="sat_power": L,a,b,c=p; return f"{L:.3g}·(1−{a:.2g}·(C+{b:.2g})^{c:.2f})"
    if slug=="logarithmic": a,c,b=p; return f"{a:.3g}·log(C+{c:.3g})" + (f" + {b:.3g}" if abs(b)>0.01 else "")
    return "?"

FAMILIES = [
    ("power_law","Power-law",f_powerlaw,3,([0.01,0.05,0.0],[100,1.5,50],["log","lin","lin"])),
    ("shifted_pl","Shifted power-law",f_shiftedpl,3,([0.001,0.01,0.05],[100,1e6,1.5],["log","log","lin"])),
    ("sat_exp","Saturating exponential",f_satexp,3,([50,1e3,0.05],[1e6,1e12,3.0],["log","log","lin"])),
    ("sat_power","Saturating power",f_satpower,4,([50,0.1,0.1,-3],[5000,2,1000,-0.05],["log","lin","log","lin"])),
    ("logarithmic","Log-with-offset",f_log,3,([0.01,-1e3,-1000],[1000,1e8,1000],["log","lin","lin"])),
]

def p0sample(ranges, rng):
    lo,hi,kinds=ranges; out=[]
    for l,h,k in zip(lo,hi,kinds):
        out.append(float(np.exp(rng.uniform(np.log(l),np.log(h)))) if k=="log" else float(rng.uniform(l,h)))
    return out

def best_fit(func, ranges, x, y, n_starts=500, seed=42):
    lo,hi,kinds=ranges; cf=(lo,hi); rng=np.random.default_rng(seed); best=None
    try:
        de=differential_evolution(lambda p:(float(np.sum((func(x,*p)-y)**2)) if np.all(np.isfinite(func(x,*p))) else 1e30),
                                  list(zip(lo,hi)),seed=42,maxiter=150,tol=1e-9,popsize=25,polish=False,init="sobol")
        if np.isfinite(de.fun):
            try:
                popt,_=curve_fit(func,x,y,p0=list(de.x),bounds=cf,maxfev=30000)
                yh=func(x,*popt)
                if np.all(np.isfinite(yh)):
                    rss=float(np.sum((yh-y)**2)); best={"popt":popt,"rss":rss}
            except Exception: pass
    except Exception: pass
    for _ in range(n_starts):
        try:
            popt,_=curve_fit(func,x,y,p0=p0sample(ranges,rng),bounds=cf,maxfev=30000)
            yh=func(x,*popt)
            if not np.all(np.isfinite(yh)): continue
            rss=float(np.sum((yh-y)**2))
            if best is None or rss<best["rss"]: best={"popt":popt,"rss":rss}
        except Exception: continue
    return best

def loocv_rmse(func, ranges, x, y, popt):
    """Leave-one-out CV RMSE, each fold seeded from full-data popt (fast single fit)."""
    lo,hi,kinds=ranges; cf=(lo,hi); n=len(x); errs=[]
    for i in range(n):
        xt=np.delete(x,i); yt=np.delete(y,i)
        try:
            pp,_=curve_fit(func,xt,yt,p0=list(popt),bounds=cf,maxfev=10000)
        except Exception:
            pp=popt
        yp=func(np.array([x[i]]),*pp)[0]
        if np.isfinite(yp): errs.append((yp-y[i])**2)
    return float(np.sqrt(np.mean(errs))) if errs else float("nan")

def runs_test_p(res):
    """Wald-Wolfowitz runs test on residual signs. Returns two-sided p."""
    signs=np.sign(res); signs=signs[signs!=0]
    n1=int(np.sum(signs>0)); n2=int(np.sum(signs<0))
    if n1==0 or n2==0: return float("nan")
    runs=1+int(np.sum(signs[1:]!=signs[:-1]))
    mu=1+2*n1*n2/(n1+n2)
    var=(2*n1*n2*(2*n1*n2-n1-n2))/(((n1+n2)**2)*(n1+n2-1))
    if var<=0: return float("nan")
    z=(runs-mu)/np.sqrt(var)
    return float(2*(1-stats.norm.cdf(abs(z))))

def durbin_watson(res):
    return float(np.sum(np.diff(res)**2)/np.sum(res**2))

# ---- compute everything ----
ALLDATA={}
for name,(model,N,dirs) in RUNS.items():
    ALLDATA[name]=load(dirs,N)

print("# Goodness-of-fit: all runs x all families")
print("# NOTE: known-error chi-square OMITTED — needs seed replicates (SEM) we don't have.\n")

for name,(model,N,dirs) in RUNS.items():
    x,y=ALLDATA[name]; n=len(x); em=y>10
    print("="*200)
    print(f"## {name}   (n={n}, L_end={int(y.max())}, n[L>10]={int(em.sum())})")
    print("="*200)
    rows=[]
    for slug,disp,func,k,ranges in FAMILIES:
        b=best_fit(func,ranges,x,y)
        if b is None:
            rows.append((disp,k,None)); continue
        popt=b["popt"]; yh=func(x,*popt); res=y-yh
        rss=float(np.sum(res**2)); rmse=float(np.sqrt(rss/n)); maxr=float(np.max(np.abs(res)))
        tss=float(np.sum((y-y.mean())**2)); r2=1-rss/tss
        yhe=func(x[em],*popt); r2e=1-np.sum((y[em]-yhe)**2)/np.sum((y[em]-y[em].mean())**2) if em.sum()>1 else float("nan")
        aic=2*k+n*np.log(rss/n); denom=n-k-1; aicc=aic+(2*k*(k+1)/denom if denom>0 else 0); bic=k*np.log(n)+n*np.log(rss/n)
        loo=loocv_rmse(func,ranges,x,y,popt)
        dw=durbin_watson(res); rp=runs_test_p(res)
        try: shp=float(stats.shapiro(res)[1]) if n>=3 else float("nan")
        except Exception: shp=float("nan")
        rows.append((disp,k,dict(eq=eqn(slug,popt),n=n,dof=n-k,rss=rss,rmse=rmse,maxr=maxr,r2=r2,r2e=r2e,
                                 aicc=aicc,bic=bic,loo=loo,dw=dw,rp=rp,shp=shp)))
    # dAICc + AICc weights within this run
    aiccs=[r[2]["aicc"] for r in rows if r[2]]
    amin=min(aiccs)
    ws=np.exp(-0.5*(np.array(aiccs)-amin)); ws=ws/ws.sum()
    wi=0
    hdr=f"{'family':<22}{'k':>3}{'RSS':>10}{'RMSE':>8}{'max|r|':>8}{'R2':>9}{'R2>10':>9}{'AICc':>9}{'dAICc':>8}{'wAICc':>8}{'BIC':>9}{'LOOCV':>8}{'DW':>6}{'runsP':>7}{'shapP':>7}"
    print(hdr); print("-"*len(hdr))
    # sort by AICc
    order=sorted(range(len(rows)), key=lambda i: rows[i][2]["aicc"] if rows[i][2] else 1e18)
    wmap={}
    j=0
    for i,r in enumerate(rows):
        if r[2]: wmap[i]=ws[j]; j+=1
    for i in order:
        disp,k,d=rows[i]
        if d is None: print(f"{disp:<22}{k:>3}  FIT FAILED"); continue
        print(f"{disp:<22}{k:>3}{d['rss']:>10.2f}{d['rmse']:>8.3f}{d['maxr']:>8.3f}{d['r2']:>9.5f}{d['r2e']:>9.5f}"
              f"{d['aicc']:>9.1f}{d['aicc']-amin:>8.1f}{wmap[i]:>8.3f}{d['bic']:>9.1f}{d['loo']:>8.3f}{d['dw']:>6.2f}{d['rp']:>7.3f}{d['shp']:>7.3f}")
    print("\nFitted equations (AICc order):")
    for i in order:
        disp,k,d=rows[i]
        if d: print(f"   {disp:<22} L = {d['eq']}")
    print()
