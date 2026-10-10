"""Recompute benchmark numbers from archived per-seed metrics, without ML/data."""
import csv
import json
import math
from pathlib import Path
import statistics
from single_object_artifacts import PACKAGE, ROOT, verified_members

MAIN = ['e2v4f_or', 'e1v4f_ft', 'e3v2f_ft', 'mlp_ctf_if', 'v3f_or']
BANDS = [(10,25),(25,40),(40,60),(60,100),(100,120)]


def summarize():
    _, members = verified_members()
    counts = json.loads((PACKAGE/'test_counts.json').read_text())
    registry = json.loads((PACKAGE/'benchmark_map.json').read_text())
    rows = []
    for key, model in registry.items():
        runs = model['runs']
        if sorted(r['seed'] for r in runs) != [42,123,456]:
            if key in MAIN:
                raise ValueError(f'Main benchmark {key} lacks three distinct seeds')
            print(f'Not reconstructed: {key} (missing three-seed evidence)')
            continue
        by_seed=[]
        for run in runs:
            m=json.loads(members['metrics/'+run['metrics']])
            n=counts['counts_by_seed'][str(run['seed'])]
            t=m['turn_on_curve']; edges=t['bins']; e=t['binned_efficiencies']
            if edges != counts['bins'] or m['calibration_split']!='test_recalibrated':
                raise ValueError(f'Metric convention differs: {key}')
            bands=[]
            for a,b in BANDS:
                indices=[i for i,x in enumerate(edges[:-1]) if a<=x<b]
                bands.append(100*sum(e[i]*n[i] for i in indices)/sum(n[i] for i in indices))
            by_seed.append({'bands':bands,'global':100*m['global_efficiency']['0.5%'],
                            'fpr':m['achieved_fake_rates']['0.5%']})
        values=[r['global'] for r in by_seed]
        row={'key':key,'label':model['label'],'parameters':model['params'],
             **{f'{a}-{b}':statistics.mean(r['bands'][i] for r in by_seed) for i,(a,b) in enumerate(BANDS)},
             'global_mean':statistics.mean(values),'global_SE':statistics.stdev(values)/math.sqrt(3),
             'min_achieved_FPR':min(r['fpr'] for r in by_seed),'max_achieved_FPR':max(r['fpr'] for r in by_seed)}
        rows.append(row)
    return rows


if __name__=='__main__':
    rows=summarize()
    out=ROOT/'results/generated/single_object/summary.csv'; out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    for key in MAIN:
        r=next(r for r in rows if r['key']==key)
        print(f"{key}: global {r['global_mean']:.2f} +/- {r['global_SE']:.2f}; bands " + ', '.join(f'{r[f"{a}-{b}"]:.2f}' for a,b in BANDS))
    print(out)
