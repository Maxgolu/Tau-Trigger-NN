import numpy as np, json, glob, os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
BASE=str(Path(os.environ.get("TAU_EXPERIMENTS", ROOT/'results/generated/single_object/metrics')).resolve()).replace('\\','/').rstrip('/')+'/'
EDGES=np.linspace(10,120,45)
BANDS=[(10,25),(25,40),(40,60),(60,100),(100,120)]
def counts(seed):
    path=Path(os.environ.get('TAU_COUNTS', ROOT/'results/single_object/test_counts.json'))
    data=json.loads(path.read_text(encoding='utf-8'))
    if not np.array_equal(data['bins'],EDGES): raise ValueError('count binning mismatch')
    return np.asarray(data['counts_by_seed'][str(seed)])
_C={}
def C(seed):
    if seed not in _C: _C[seed]=counts(seed)
    return _C[seed]
def seed_of(path):
    path=str(path).replace('\\','/')
    for s in (42,123,456):
        if f"_s{s}_" in path or f"_s{s}/" in path: return s
    raise ValueError(path)
def bands(eff, seed, mode='pooled'):
    n=C(seed); lo=EDGES[:-1]; out=[]
    for a,b in BANDS:
        m=(lo>=a)&(lo<b)
        if mode=='pooled': out.append(np.sum(eff[m]*n[m])/np.sum(n[m]))
        else: out.append(np.mean(eff[m]))
    return np.array(out)
def load(path):
    m=json.load(open(path)); s=seed_of(path)
    if m.get('calibration_split') != 'test_recalibrated': raise ValueError(f'wrong calibration: {path}')
    if not np.array_equal(m['turn_on_curve']['bins'],EDGES): raise ValueError(f'wrong bins: {path}')
    e=np.array(m['turn_on_curve']['binned_efficiencies']); b=np.array(m['baseline_tob_pt']['binned_efficiencies'])
    return dict(seed=s, eff=e, err=np.array(m['turn_on_curve']['binned_efficiencies_err']), base=b, base_err=np.array(m['baseline_tob_pt']['binned_efficiencies_err']),
                glob=m['global_efficiency']['0.5%'], base_glob=m['baseline_tob_pt']['global_efficiency'],
                fpr=m['achieved_fake_rates']['0.5%'], base_fpr=m['baseline_tob_pt']['achieved_fake_rate'], m=m)
def summarize(paths, mode='pooled'):
    rs=[load(p) for p in paths]
    if sorted(r['seed'] for r in rs) != [42,123,456]: raise ValueError('Require exactly one run per seed: 42, 123, 456')
    B=np.array([bands(r['eff'],r['seed'],mode) for r in rs]); G=np.array([r['glob'] for r in rs])
    BB=np.array([bands(r['base'],r['seed'],mode) for r in rs]); GB=np.array([r['base_glob'] for r in rs])
    return dict(bands=B.mean(0)*100, glob=G.mean()*100, glob_se=G.std(ddof=1)/np.sqrt(len(G))*100 if len(G)>1 else 0, glob_sd=G.std(ddof=1)*100 if len(G)>1 else 0,
                base_bands=BB.mean(0)*100, base_glob=GB.mean()*100, seeds=[r['seed'] for r in rs], fpr=[r['fpr'] for r in rs], rs=rs)
