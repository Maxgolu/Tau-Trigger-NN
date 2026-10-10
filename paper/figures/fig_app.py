import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
import matplotlib.font_manager as fm
for _f in ["regular","italic","bold"]:
    try: fm.fontManager.addfont(f"/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-{_f}.otf")
    except Exception: pass  # falls back to the default serif font
mpl.rcParams.update({"font.family":"serif","font.serif":["TeX Gyre Termes","Times New Roman","DejaVu Serif"],"font.size":7,"axes.linewidth":0.5,"mathtext.fontset":"stix",
 "axes.edgecolor":"#52514e","xtick.color":"#52514e","ytick.color":"#52514e","xtick.major.width":0.5,"ytick.major.width":0.5,"xtick.major.size":2.5,"ytick.major.size":2.5})
TAU="#2a78d6"; BKG="#eb6834"
df=pd.read_pickle('obj.pkl'); tr=df[df.split=='train']
bands=[(5,10),(10,20),(20,40),(40,80),(80,120)]
fig,axs=plt.subplots(1,5,figsize=(6.75,1.45),sharey=False)
fb=np.linspace(0.2,1.0,33)
for ax,(lo,hi) in zip(axs,bands):
    m=(tr.tob_pt_gev>=lo)&(tr.tob_pt_gev<hi)
    for lab,sel,c in [("Truth $\\tau$",tr.is_tau==1,TAU),("Non-$\\tau$",tr.is_tau==0,BKG)]:
        x=tr[m&sel].frac; h,_=np.histogram(x,bins=fb,density=True); ax.stairs(h,fb,color=c,lw=1.0,label=lab,baseline=None)
    ax.set_title(f"{lo}–{hi} GeV",fontsize=7); ax.set_xlim(0.2,1); ax.set_xticks([0.2,0.6,1.0]); ax.spines[['top','right']].set_visible(False)
    ax.set_xlabel("$\\mathrm{Frac}_{\\mathrm{EM2}}$",fontsize=6.5); ax.tick_params(labelsize=6)
axs[0].set_ylabel("Density"); axs[2].legend(frameon=False,fontsize=5.8,loc='upper left',handlelength=1.2)
fig.tight_layout(pad=0.3,w_pad=0.6); fig.savefig('fig_frac_conditioned.pdf'); fig.savefig('fig_frac_conditioned.png',dpi=200)
