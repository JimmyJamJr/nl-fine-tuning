"""Figures 7 and 8 as one two-panel figure: (a) Qwen3-0.6B step-size sweep, (b) Pythia family scaling (s=1),
each half text-width and taller than before; panel (a) lines semi-transparent so overlapping runs stay visible.
Both panels use the completed-stage convention."""
import os, re, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
try:
    import seaborn  # registers mako/rocket/crest/flare colormaps
except Exception:
    pass
from matplotlib.colors import LinearSegmentedColormap
for _n,_c in {"blue_red": ["#08306b","#2166ac","#7b52ab","#c0392b","#7f0000"],     # blue -> purple -> red, no pale middle
              "blue_red2": ["#1f4e9c","#4a4fa8","#8b3fa0","#c2453f","#8f1010"]}.items():  # slightly softer path
    if _n not in plt.colormaps: matplotlib.colormaps.register(LinearSegmentedColormap.from_list(_n,_c))
from matplotlib.ticker import FuncFormatter
SCRATCH="/scratch/gautschi/huan2073"; OUT="/home/huan2073/nl-fine-tuning/nl/plots/paper_fig5"; os.makedirs(OUT, exist_ok=True)
# ---- (a) step-size sweep: reuse the paper script's chains and completed-stage loader ----
src=open("/home/huan2073/nl-fine-tuning/nl/plot_stepsize_sweep.py").read()
ns={}; exec(compile(src.split("kfmt=FuncFormatter")[0], "stepsize_head", "exec"), ns)
CHAINS, load_s, F, CAP7, TARGET = ns["CHAINS"], ns["load"], ns["F"], ns.get("CAP",300_000), ns.get("TARGET",256)
import sys; sys.path.insert(0,"/scratch/gautschi/huan2073/audit_tmp/completed"); from chain_cache import cache_wrap; load_s=cache_wrap(load_s,"stepsize")
# ---- (b) family scaling: completed stage per step ----
RUNS={"Pythia-160M":(162_322_944,"#440154",["jackie_pythia160m_step1_lr1e4_resumed","12587988","12854034","12969577","13364117","13542861","13606486","13930099","14249842","11590957"]),
      "Pythia-410M":(405_334_016,"#3b528b",["10696605","10730890","11114263","11426007","12494239","13930098","14110833"]),
      "Pythia-1.4B":(1_414_647_808,"#5ec962",["10579766","10682439","10730893","11696279","15232695","11650850"]),
      "Pythia-2.8B":(2_775_208_960,"#b5a800",["10580483","10694584","10730892","11426089","11647896"])}
def load_f(dirs,N):
    factor=6*N/1e15; seen=-1; cum=0; pf=[]; Ls=[]; prev=None; prevL=None; done=0
    for d in dirs:
        p=f"{SCRATCH}/nl_output/search/job_{d}/loss_history.jsonl"
        if not os.path.exists(p): continue
        for line in open(p):
            try: e=json.loads(line)
            except: continue
            s=e["step"]
            if s<=seen: continue
            seen=s; cum+=e.get("tokens",0); L=e.get("effective_L"); st=e.get("stage")
            if L is None or st is None: continue
            if prev is not None and st!=prev: done=prevL
            prev=st; prevL=L
            pf.append(cum*factor); Ls.append(done)
    return np.array(pf), np.array(Ls)
CAP8=1_000_000
fmt=FuncFormatter(lambda v,_: "0" if v==0 else (f"{v/1e6:g}M" if v>=1e6 else f"{v/1e3:g}k"))
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10.6,4.9))
CMA=plt.get_cmap(os.environ.get('CMA_NAME','plasma')); CMB=plt.get_cmap(os.environ.get('CMB_NAME','inferno')); SUF=os.environ.get('OUT_SUFFIX','')
SVALS=sorted(CHAINS); COLA={s_:CMA(float(os.environ.get('CMA_LO','0.06'))+float(os.environ.get('CMA_SPAN','0.86'))*i/(len(SVALS)-1)) for i,s_ in enumerate(SVALS)}
_sat=float(os.environ.get('SAT_A','1.0'))
if _sat!=1.0:   # optional saturation boost for low-chroma maps (e.g. cividis)
    from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
    for _k,_c in list(COLA.items()):
        _h=rgb_to_hsv(np.array(_c[:3])); _h[1]=min(1.0,_h[1]*_sat); COLA[_k]=(*hsv_to_rgb(_h),1.0)
ALPHA_A=float(os.environ.get('ALPHA_A','0.72')); ALPHA_B=float(os.environ.get('ALPHA_B','0.72'))
NAMES=list(RUNS); COLB={n_:CMB(0.15+0.65*i/(len(NAMES)-1)) for i,n_ in enumerate(NAMES)}
# (a)
for s,(col,dirs) in CHAINS.items():
    pf,Ls,end_cum=load_s(dirs,s); cleared=int(Ls.max())
    if cleared+s>=TARGET and cleared<TARGET: pf=np.append(pf,end_cum); Ls=np.append(Ls,TARGET)
    m=pf<=CAP7; pfc,Lsc=pf[m],Ls[m]
    if pfc[-1]<CAP7: pfc=np.append(pfc,CAP7); Lsc=np.append(Lsc,Lsc[-1])
    ax1.plot(pfc,Lsc,"-",color=COLA[s],lw=1.7,alpha=ALPHA_A,label=f"$s={s}$",drawstyle="steps-post")
ax1.set_xlim(0,CAP7); ax1.set_ylim(0,268); ax1.set_xticks(np.arange(0,CAP7+1,100_000)); ax1.xaxis.set_major_formatter(fmt)
ax1.set_yticks(np.arange(0,257,32))
ax1.set_xlabel("Cumulative Compute (PFLOPs)",fontsize=11); ax1.set_ylabel("Achieved Lookahead $L$",fontsize=11)
ax1.set_title("Achieved Lookahead vs Cumulative Compute\nQwen3-0.6B step size ($s$) sweep",fontsize=11); ax1.grid(True,alpha=0.3)
ax1.legend(loc="upper left",fontsize=9,ncol=2,framealpha=0.9,title="step size",title_fontsize=9)
load_f=cache_wrap(load_f,"family")
# (b)
for name,(N,c,dirs) in RUNS.items():
    pf,Ls=load_f(dirs,N); full_max=pf.max(); m=(pf<=CAP8)&(Ls>=1); pf,Ls=pf[m],Ls[m]
    if len(pf)>4000: idx=np.linspace(0,len(pf)-1,4000).astype(int); pf,Ls=pf[idx],Ls[idx]
    # Hold the last completed stage out to the cap. Runs that continue past CAP8 are known to sit at
    # that level throughout. Pythia-160M stops just short (995,723 PFLOPs) with stage 10 in progress:
    # that stage has already consumed ~74K PFLOPs without completing, so the remaining ~4K cannot
    # change the level either. Extending it avoids a curve that appears to end early.
    if pf[-1]<CAP8: pf=np.append(pf,CAP8); Ls=np.append(Ls,Ls[-1])
    ax2.plot(pf,Ls,"-",color=COLB[name],lw=1.9,alpha=ALPHA_B,label=name,drawstyle="steps-post"); print(f"{name}: L at end/cap = {int(Ls[-1])} at {pf[-1]:,.0f}")
ax2.set_xlim(-CAP8*0.02,CAP8); ax2.set_ylim(0,105); ax2.set_xticks(np.arange(0,CAP8+1,250_000)); ax2.xaxis.set_major_formatter(fmt)
ax2.set_xlabel("Cumulative Compute (PFLOPs)",fontsize=11); ax2.set_ylabel("Achieved Lookahead $L$",fontsize=11)
ax2.set_title("Achieved Lookahead vs Cumulative Compute\nPythia family scaling ($s{=}1$)",fontsize=11); ax2.grid(True,alpha=0.3); ax2.legend(loc="upper left",fontsize=9,framealpha=0.9)
for ax in (ax1,ax2): ax.tick_params(labelsize=9.5)
fig.tight_layout(w_pad=2.0)
fig.savefig(f"{OUT}/stepsize_and_family_scaling{SUF}.png",dpi=170,bbox_inches="tight"); fig.savefig(f"{OUT}/stepsize_and_family_scaling{SUF}.pdf",bbox_inches="tight")
print("saved", f"{OUT}/stepsize_and_family_scaling{SUF}.(png,pdf)")
