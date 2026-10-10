import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
import matplotlib.font_manager as fm
for _f in ["regular","italic","bold"]:
    try: fm.fontManager.addfont(f"/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-{_f}.otf")
    except Exception: pass  # falls back to the default serif font
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
mpl.rcParams.update({"font.family":"serif","font.serif":["TeX Gyre Termes","Times New Roman","DejaVu Serif"],"font.size":7,"axes.linewidth":0.5,"mathtext.fontset":"stix"})
df=pd.read_pickle('obj.pkl'); T=np.load('T.npy'); E=np.load('E.npy')
blues=["#fcfcfb","#cde2fb","#9ec5f4","#6da7ec","#3987e5","#256abf","#184f95","#0d366b"]
cm=LinearSegmentedColormap.from_list("b",blues)
tr=df[df.split=='train']
m=(tr.tob_pt_gev>=28)&(tr.tob_pt_gev<34)
tau=tr[m&(tr.is_tau==1)&(tr.prongs==1)]; bkg=tr[m&(tr.is_tau==0)&(tr.src=='bkg')]
df['ratio']=T.sum(axis=(1,2,3))/1000/df.tob_pt_gev
tau=df.loc[tau.index]; bkg=df.loc[bkg.index]
def pick(sub):
    rf=sub.frac.rank(pct=True); rr=sub.ratio.rank(pct=True)
    d=(rf-0.5)**2+(rr-0.5)**2; return d.sort_values().index[0]
it,ib=pick(tau),pick(bkg)
print(df.loc[it,['tob_pt_gev','truth_pt_gev','frac','prongs']].to_dict(), df.loc[ib,['tob_pt_gev','frac']].to_dict())
layers=["EM0 (presampler)","EM1 (front/strip)","EM2 (middle)","EM3 (back)","HAD"]
fig=plt.figure(figsize=(6.75,2.5))
gs=fig.add_gridspec(2,7,width_ratios=[1,1,1,1,1,0.05,1.25],wspace=0.16,hspace=0.35,left=0.075,right=0.955,top=0.86,bottom=0.05)
for r,(idx,name) in enumerate([(it,"Truth-matched $\\tau$"),(ib,"Background (QCD jet)")]):
    t=T[idx]/1000; e=E[idx]/1000; vmax=max(t.max(), 1e-9)
    for L in range(5):
        ax=fig.add_subplot(gs[r,L]); ax.imshow(t[L],cmap=cm,vmin=0,vmax=vmax)
        for a in range(3):
            for b in range(3):
                v=t[L,a,b]; ax.text(b,a,f"{v:.1f}" if v>=0.05 else "0",ha='center',va='center',fontsize=5.2,color="white" if v>0.55*vmax else "#0b0b0b")
        ax.set_xticks([]); ax.set_yticks([])
        if r==0: ax.set_title(layers[L],fontsize=6.5)
        if L==0: ax.set_ylabel(name+f"\n$p_T^{{TOB}}$={df.loc[idx,'tob_pt_gev']:.1f} GeV",fontsize=6.5)
    ax=fig.add_subplot(gs[r,6]); im=ax.imshow(e,cmap=cm,vmin=0,vmax=e.max())
    for k in range(1,3):
        ax.axhline(4*k-0.5,color="#52514e",lw=0.6,ls=(0,(2,1.5))); ax.axvline(4*k-0.5,color="#52514e",lw=0.6,ls=(0,(2,1.5)))
    ax.set_xticks([]); ax.set_yticks([])
    if r==0: ax.set_title("EM2, 12$\\times$12 cells",fontsize=6.5)
    cb=fig.colorbar(im,ax=ax,fraction=0.05,pad=0.03); cb.ax.tick_params(labelsize=5,width=0.4,length=2); cb.locator=mpl.ticker.MaxNLocator(4); cb.update_ticks(); cb.ax.set_title("GeV",fontsize=5.5,pad=2)
fig.savefig('fig_calo_input.pdf'); fig.savefig('fig_calo_input.png',dpi=200)
