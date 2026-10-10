"""Leaderboards for Appendix I, recomputed from every staged three-seed configuration."""
import os, sys, json, re
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scan, lib
import make_results as mr   # style, paths, helpers

BASE = mr.BASE_BANDS
MANUAL = {  # groups without a staged config: (arch, scalars, obj, clf, params)
    'cnn_v8_or_conv2x2_bce': ('V2', 'F', 'BCE', 'OR', 2421), 'cnn_v8_or_conv2x2_invfreq': ('V2', 'F', 'IF', 'OR', 2421),
    'cnn_v8_or_dual_bce': ('V4', 'F', 'BCE', 'OR', 5973), 'cnn_v8_or_dual_invfreq': ('V4', 'F', 'IF', 'OR', 5973),
    'cnn_v8_or_mix1x1_bce': ('V1', 'F', 'BCE', 'OR', 4161), 'cnn_v8_or_mix1x1_invfreq': ('V1', 'F', 'IF', 'OR', 4161),
    'cnn_v8_or_serial_bce': ('V3', 'F', 'BCE', 'OR', 2853), 'cnn_v8_or_serial_invfreq': ('V3', 'F', 'IF', 'OR', 2853),
    'constrained_stage_g_or/run_g001': ('MLP', 'CT+F', 'FT', 'OR b=0.5‰', 2049),
    'constrained_stage_g_or/run_g002': ('MLP', 'CT+F', 'FT', 'OR b=1‰', 2049),
    'constrained_stage_g_or/run_g003': ('MLP', 'CT+F', 'FT', 'OR b=1.5‰', 2049),
    'constrained_stage_j_rankor/run_j001': ('MLP', 'CT+F', 'FT (rank)', 'OR', 2049),
    'cnn_v14_em2stride_sum': ('E3', 'F+SUM', 'IF', 'NN', 4089),
    'constrained_stage_h_rank/run_h001': ('MLP', 'CT+F', 'p=-1+FT, rank control', 'NN', 2049),
    'constrained_stage_h_rank/run_h002': ('MLP', 'CT+F', 'p=-1+FT (rank variant 2)', 'NN', 2049),
    'constrained_stage_h_rank/run_h003': ('MLP', 'CT+F', 'p=-1+FT (rank variant 3)', 'NN', 2049),
}

def tex_name(d):
    arch, sc, obj, clf, _ = d
    flat = sc.startswith('flat')
    parts = [p for p in sc.replace('flat ', '').split('+') if p]
    mp = {'CT': r'\CT', 'F': r'\Ffeat', 'SUM': r'\SUM', 'CP': r'\CP'}
    order = ['CP', 'CT', 'SUM', 'F']
    parts = sorted(parts, key=lambda p: order.index(p) if p in order else 9)
    sct = '+'.join(mp.get(p, p) for p in parts)
    obj = obj.replace('a=0.0', 'BCE').replace('p=-1.0', '$p{=}-1$').replace('p=-1', '$p{=}-1$')
    if arch == 'MLP':
        name = f'MLP, {sct}, {obj}'
    elif flat:
        name = f'{arch}, flat {sct}, {obj}'
    elif sct and arch.startswith('E') and '+' not in arch:
        name = f'{arch}, {sct}, {obj}'
    elif sct:
        name = f'{arch}+{sct}, {obj}'
    elif '+' in arch:
        name = f'{arch}, {obj}'
    else:
        name = f'{arch} (\\CT), {obj}'
    if clf != 'NN':
        name += ', ' + clf.replace('‰', r'\textperthousand')
    return name

def rows():
    out = []
    seen = set()
    for r in scan.scan():
        if r['n'] < 3: continue
        key = (round(r['glob'], 3), tuple(np.round(r['bands'], 3)))
        if key in seen: continue          # the same runs stored twice
        seen.add(key)
        d = r['desc']
        mk = [v for k, v in MANUAL.items() if r['group'].startswith(k)]
        if mk: d = None
        if d is None:
            m = [v for k, v in MANUAL.items() if r['group'].startswith(k)]
            d = m[0] if m else from_name(r['group'])
            if d is None: print('no description for', r['group']); continue
        else:
            d = (d['arch'], d['scalars'], d['obj'], d['clf'], d['params'])
        out.append(dict(group=r['group'], d=d, name=tex_name(d), bands=r['bands'], glob=r['glob'], se=r['se'],
                        params=d[4], paths=r['paths']))
    return out

EP = {'single': ('E1', 48, 100), 'ladder': ('E2', 696, 72), 'stride': ('E3', 696, 72)}
VP = {'mix1x1': ('V1', 96, 108), 'conv2x2': ('V2', 276, 48), 'serial': ('V3', 708, 48), 'dual': ('V4', 372, 156),
      'dualbranch': ('V4', 372, 156)}

def from_name(g):
    r = g.split('/')[-1].replace('run_', '')
    m = re.match(r'v1[13]_(ft_|or_)?em2(single|ladder|stride)_(mix1x1|conv2x2|dual)(_frac)?(_bce)?$', r)
    if m:
        ft, e, v, fr, bce = m.groups()
        E, pe, oe = EP[e]; V, pv, ov = VP[v]; f = 1 if fr or ft == 'or_' else 0
        params = pe + pv + 32 * (oe + ov + f) + 577
        obj = 'BCE' if (bce or ft == 'or_') else ('FT' if ft == 'ft_' else 'IF')
        return (f'{E}+{V}', 'F' if f else '', obj, 'OR' if ft == 'or_' else 'NN', params)
    m = re.match(r'v9_em2(single|ladder|stride)_(mix1x1|conv2x2|dual|serial)$', r)
    if m:
        e, v = m.groups(); E, pe, oe = EP[e]; V, pv, ov = VP[v]
        return (f'{E}+{V}', '', 'IF', 'NN', pe + pv + 32 * (oe + ov) + 577)
    m = re.match(r'v10_ft_em2(single|ladder|stride)_(mix1x1|conv2x2|dual|serial)$', r)
    if m:
        e, v = m.groups(); E, pe, oe = EP[e]; V, pv, ov = VP[v]
        return (f'{E}+{V}', '', 'FT', 'NN', pe + pv + 32 * (oe + ov) + 577)
    m = re.match(r'v10_ft_(mix1x1|conv2x2|serial|dual)_frac$', r)
    if m:
        V, pv, ov = VP[m.group(1)]
        return (V, 'F', 'FT', 'NN', pv + 32 * (ov + 1) + 577)
    m = re.match(r'v[67]_(mix1x1|conv2x2|serial|dual|dualbranch)_(raw|frac)$', r)
    if m:
        v, t = m.groups(); V, pv, ov = VP[v]; f = 1 if t == 'frac' else 0
        return (V, 'F' if f else '', 'IF', 'NN', pv + 32 * (ov + f) + 577)
    return None

def conv_ct(d): return any(p.startswith('V') for p in d[0].split('+'))

LISTS = {
    'balanced': lambda R: [r for r in R if r['bands'][4] >= 98.5 and not (conv_ct(r['d']) and r['params'] >= 6000)][:5],
    'plateau':  lambda R: [r for r in R if r['bands'][4] >= 99.0][:5],
    'top':      lambda R: R[:10],
    'budget':   lambda R: [r for r in R if r['params'] <= 5000][:5],
}

def write(R):
    R = sorted(R, key=lambda r: -r['glob'])
    res = {}
    for name, f in LISTS.items():
        sel = f(R); res[name] = sel
        lines = []
        for r in sel:
            st = dict(bands=r['bands'], glob=r['glob'], se=r['se'], src='metrics')
            p = mr.kparams(r['params']) + (r'$^\dagger$' if name == 'top' and r['params'] > 6000 else '')
            lines.append(mr.row(r['name'], st, p))
        lines.append(r'\midrule')
        lines.append(r'eFEX baseline & 2.28 & 35.85 & 81.60 & 96.89 & 99.34 & 57.37 & --- \\')
        mr.write_tabular(f'tab_lb_{name}.tex', lines)
    return res

PAL = ['#2a78d6', '#eb6834', '#2e9e5b', '#8e5bd6', '#c9a227', '#6b6b6b', '#e34948', '#1b9aaa', '#a0522d', '#d67fb0']

def plain(n):
    return (n.replace(r'\Ffeat', 'F').replace(r'\CT', 'CT').replace(r'\SUM', 'SUM').replace(r'\CP', 'CP')
             .replace('$p{=}-1$', 'p=-1').replace(r'\textperthousand', '‰'))

W10 = [(60, 70), (70, 80), (80, 90), (90, 100), (100, 110), (110, 120)]

def pooled_gain(rs):
    lo = lib.EDGES[:-1]; D = []; E = []
    for x in rs:
        n = lib.C(x['seed']); dd = []; ee = []
        for a, b in W10:
            m = (lo >= a) & (lo < b); N = n[m].sum()
            em = np.sum(x['eff'][m] * n[m]) / N; eb = np.sum(x['base'][m] * n[m]) / N
            dd.append(em - eb); ee.append(np.sqrt(em * (1 - em) / N))
        D.append(dd); E.append(ee)
    D = np.array(D) * 100; E = np.array(E) * 100
    return np.array([(a + b) / 2 for a, b in W10]), D.mean(0), np.sqrt((E ** 2).sum(0)) / len(rs)

def figure(name, sel):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.75, 2.1), gridspec_kw=dict(width_ratios=[1.35, 1]))
    C = mr.CENT; base = None
    for j, r in enumerate(sel):
        rs = [lib.load(p) for p in r['paths']]
        e = np.mean([x['eff'] for x in rs], 0); er = np.sqrt(np.sum([x['err'] ** 2 for x in rs], 0)) / len(rs)
        b = np.mean([x['base'] for x in rs], 0)
        if base is None:
            base = b; a1.plot(C, b, color='black', lw=1.0, label='eFEX baseline')
        c = PAL[j % len(PAL)]
        a1.plot(C, e, color=c, lw=0.8, marker='o', ms=1.1, label=plain(r['name']))
        xs, ds, es = pooled_gain(rs)
        off = (j - (len(sel) - 1) / 2) * 0.55
        a2.errorbar(xs + off, ds, yerr=es, color=c, lw=0, elinewidth=0.6, marker='o', ms=2.0, capsize=0)
    a2.axhline(0, color='black', lw=0.8)
    a1.set_xlim(10, 120); a1.set_ylim(0, 1.02); a1.set_xlabel('True $p_T$ [GeV]'); a1.set_ylabel('Signal efficiency')
    a1.legend(frameon=False, fontsize=5.2, loc='lower right', handlelength=1.4)
    a2.set_xlim(60, 120); a2.set_xlabel('True $p_T$ [GeV]'); a2.set_ylabel('Gain over baseline [points]')
    a2.set_title('saturation region, 10-GeV windows', fontsize=6.5, pad=2)
    for a in (a1, a2):
        a.spines[['top', 'right']].set_visible(False); a.grid(alpha=0.25, lw=0.3); a.tick_params(labelsize=6)
    fig.tight_layout(pad=0.3, w_pad=1.2)
    fig.savefig(f'{mr.PAPER}/figures/fig_lb_{name}.pdf'); fig.savefig(f'{mr.PAPER}/figures/fig_lb_{name}.png', dpi=200)
    plt.close(fig)

if __name__ == '__main__':
    R = rows()
    res = write(R)
    for k, v in res.items(): figure(k, v)
    for k, v in res.items():
        print('==', k)
        for r in v: print(f"  {r['glob']:.2f} {r['name']:45s} {np.round(r['bands'],2)} {r['params']}")
    json.dump({k: [r['group'] for r in v] for k, v in res.items()}, open(f'{mr.PAPER}/lb_groups.json', 'w'), indent=1)
