import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
import matplotlib.font_manager as fm
for _f in ["regular","italic","bold"]:
    try: fm.fontManager.addfont(f"/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-{_f}.otf")
    except Exception: pass  # falls back to the default serif font
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from numpy.lib.stride_tricks import sliding_window_view
mpl.rcParams.update({"font.family":"serif","font.serif":["TeX Gyre Termes","Times New Roman","DejaVu Serif"],"font.size":7,"axes.linewidth":0.5,"mathtext.fontset":"stix",
 "axes.edgecolor":"#52514e","xtick.color":"#52514e","ytick.color":"#52514e","axes.labelcolor":"#0b0b0b","xtick.major.width":0.5,"ytick.major.width":0.5,"xtick.major.size":2.5,"ytick.major.size":2.5,"xtick.minor.size":1.5,"ytick.minor.size":1.5})
TAU="#2a78d6"; BKG="#eb6834"; INK2="#52514e"
blues=["#fcfcfb","#cde2fb","#9ec5f4","#6da7ec","#3987e5","#256abf","#184f95","#0d366b"]
cm=LinearSegmentedColormap.from_list("b",blues)
df=pd.read_pickle('obj.pkl'); E=np.load('E.npy'); T=np.load('T.npy')
tr=df[df.split=='train']
fig=plt.figure(figsize=(6.75,1.95))
gs=fig.add_gridspec(1,4,width_ratios=[1.25,1.25,0.8,0.8],wspace=0.42,left=0.075,right=0.99,top=0.86,bottom=0.2)
# (a) tob_pt
ax=fig.add_subplot(gs[0]); bins=np.arange(5,120.5,2.5)
for lab,sel,c in [("Truth $\\tau$",tr.is_tau==1,TAU),("Non-$\\tau$ object",tr.is_tau==0,BKG)]:
    x=tr[sel].tob_pt_gev; h,_=np.histogram(x[(x>=5)&(x<120)],bins=bins,density=True); ax.stairs(h,bins,lw=1.1,color=c,label=lab,baseline=None)
ax.set_yscale('log'); ax.set_xlabel("$p_T^{\\mathrm{TOB}}$ [GeV]"); ax.set_ylabel("Normalized density"); ax.set_xlim(5,120); ax.set_xticks([5,25,50,75,100,120])
ax.legend(frameon=False,fontsize=6,loc='upper right',handlelength=1.4); ax.set_title("(a) TOB transverse momentum",fontsize=7)
ax.spines[['top','right']].set_visible(False)
# (b) fraction in band
ax=fig.add_subplot(gs[1]); lo,hi=20,40; m=(tr.tob_pt_gev>=lo)&(tr.tob_pt_gev<hi); fb=np.linspace(0.2,1.0,41)
for lab,sel,c in [("Truth $\\tau$",tr.is_tau==1,TAU),("Non-$\\tau$ object",tr.is_tau==0,BKG)]:
    x=tr[m&sel].frac; h,_=np.histogram(x,bins=fb,density=True); ax.stairs(h,fb,lw=1.1,color=c,label=lab,baseline=None)
    ax.axvline(x.median(),color=c,lw=0.7,ls=(0,(2,1.5)))
ax.set_xlabel("$\\mathrm{Frac}_{\\mathrm{EM2}}$"); ax.set_ylabel("Normalized density"); ax.set_xlim(0.2,1.0)
ax.set_title(f"(b) EM2 compactness, {lo}$\\leq p_T^{{\\mathrm{{TOB}}}}$<{hi} GeV",fontsize=7)
ax.legend(frameon=False,fontsize=6,loc='upper left',handlelength=1.4); ax.spines[['top','right']].set_visible(False)
# (c) illustration using the same representative objects as the calorimeter figure
df['ratio']=T.sum(axis=(1,2,3))/1000/df.tob_pt_gev
mm=(tr.tob_pt_gev>=28)&(tr.tob_pt_gev<34)
def pick(sub):
    sub=df.loc[sub.index]; rf=sub.frac.rank(pct=True); rr=sub.ratio.rank(pct=True)
    return ((rf-0.5)**2+(rr-0.5)**2).sort_values().index[0]
it=pick(tr[mm&(tr.is_tau==1)&(tr.prongs==1)]); ib=pick(tr[mm&(tr.is_tau==0)&(tr.src=='bkg')])
for k,(idx,lab) in enumerate([(it,"(c) $\\tau$"),(ib,"(d) jet")]):
    ax=fig.add_subplot(gs[2+k]); e=E[idx]/1000; ax.imshow(e,cmap=cm,vmin=0,vmax=e.max())
    w=sliding_window_view(e,(3,3)).sum(axis=(-1,-2)); r,c=np.unravel_index(w.argmax(),w.shape)
    ax.add_patch(Rectangle((c-0.5,r-0.5),3,3,fill=False,ec="#e34948",lw=1.1))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color(INK2)
    ax.set_title(lab,fontsize=7)
    ax.set_xlabel(f"{w.max():.1f} / {e.sum():.1f} GeV = {w.max()/e.sum():.2f}",fontsize=6.3)
fig.savefig('fig_features.pdf'); fig.savefig('fig_features.png',dpi=200)
print(tr[m&(tr.is_tau==1)].frac.median(), tr[m&(tr.is_tau==0)].frac.median())
