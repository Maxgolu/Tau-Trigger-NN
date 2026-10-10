import numpy as np, pandas as pd
import sys, os
# Usage: python prep.py "<path to Tau Trigger Project>"  (reads Signal/ and Background/)
base=os.path.join(sys.argv[1] if len(sys.argv)>1 else ".", "")
sig_csv=pd.read_csv(base+"Signal/signal_combined.csv"); bkg_csv=pd.read_csv(base+"Background/bkg_combined.csv")
S=np.load(base+"Signal/signal_combined.npz"); B=np.load(base+"Background/bkg_combined.npz")
off=2*S['event_nums'].max()
out={}
rows=[]
for tag,df,d,o in [("sig",sig_csv,S,0),("bkg",bkg_csv,B,off)]:
    ev=d['event_nums']; pos=pd.Series(np.arange(len(ev)),index=ev)
    ei=pos.loc[df.eventNumber.values].values; ti=df.tob_index.values
    T=d['X_tensors'][ei,ti]; E=d['X_em2_tensors'][ei,ti]
    out[tag]=dict(T=T,E=E)
    df=df.copy(); df['eventNumber']=df.eventNumber+o; df['src']=tag
    rows.append(df)
df=pd.concat(rows,ignore_index=True)
T=np.concatenate([out['sig']['T'],out['bkg']['T']]); E=np.concatenate([out['sig']['E'],out['bkg']['E']])
# split seed 42, 70/10/20 over unique events
u=np.unique(df.eventNumber.values).copy(); rng=np.random.RandomState(42); rng.shuffle(u)
n=len(u); tr=set(u[:int(n*.7)]); va=set(u[int(n*.7):int(n*.8)])
df['split']=np.where(df.eventNumber.isin(tr),'train',np.where(df.eventNumber.isin(va),'val','test'))
# em2 fraction
tot=E.sum(axis=(1,2))
from numpy.lib.stride_tricks import sliding_window_view
w=sliding_window_view(E,(3,3),axis=(1,2)).sum(axis=(-1,-2))
best=w.reshape(len(E),-1).max(1)
df['frac']=np.where(tot==0,0,best/np.where(tot==0,1,tot))
df['tob_pt_gev']=df.tob_pt/1000; df['truth_pt_gev']=df.truth_pt/1000
df['is_tau']=((df.Type=='Signal')&(df.signal==1)).astype(int)
np.save('T.npy',T); np.save('E.npy',E)
df[['eventNumber','tob_index','src','split','is_tau','prongs','tob_pt_gev','truth_pt_gev','tob_eta','tob_phi','tob_bdt','frac']].to_pickle('obj.pkl')
print(df.groupby('split').eventNumber.nunique(), df.groupby(['split','is_tau']).size())
print("events total",df.eventNumber.nunique(),"objects",len(df),"taus",df.is_tau.sum())
tr=df[df.split=='train']
print("train sig events",tr[tr.src=='sig'].eventNumber.nunique(),"bkg",tr[tr.src=='bkg'].eventNumber.nunique())
