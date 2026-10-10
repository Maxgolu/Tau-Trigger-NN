"""Results figures and tables for Section 5 and Appendix I.

Every number comes from the metrics.json files of the three seeds (test-recalibrated
operating point, 0.5% event FPR).  Band efficiencies pool the 2.5-GeV bins with their
test tau counts, which reproduces the chapter tables of the project report exactly.
Missing metric evidence is an error, never replaced with report constants.
The prior-work fitted panel and OR_PAIRS are explicitly historical reported
values, not newly evaluated predictions. See results/single_object/README.md.
"""
import glob, json, os, sys
import numpy as np
import matplotlib as mpl, matplotlib.pyplot as plt
import matplotlib.font_manager as fm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib

for _f in ["regular", "italic", "bold"]:
    try: fm.fontManager.addfont(f"/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-{_f}.otf")
    except Exception: pass
mpl.rcParams.update({"font.family": "serif", "font.serif": ["TeX Gyre Termes", "Times New Roman", "DejaVu Serif"],
    "font.size": 7, "axes.linewidth": 0.5, "mathtext.fontset": "stix", "axes.edgecolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "legend.fontsize": 6})

B = lib.BASE
PAPER = str(os.environ.get('TAU_PAPER', lib.ROOT/'results/generated/single_object/paper'))
os.makedirs(f'{PAPER}/tables', exist_ok=True)
os.makedirs(f'{PAPER}/figures', exist_ok=True)
CENT = 0.5 * (lib.EDGES[:-1] + lib.EDGES[1:])

# ----------------------------------------------------------------------------
# model registry: key -> (label, glob of run dirs, params, report fallback)
# report fallback = (10-25, 25-40, 40-60, 60-100, 100-120, global, global_se)
# ----------------------------------------------------------------------------
M = {
 # benchmark (Section 5.1)
 'e2v4f_or': (r'E2+V4+\Ffeat, BCE, OR', 'cnn_v13_or_bce_s*/run_v13_or_em2ladder_dual_*', 8973, None),
 'v3f_or':   (r'V3+\Ffeat, IF, OR', 'cnn_v8_or_serial_invfreq/run_*', 2853, None),
 'e1v4f_ft': (r'E1+V4+\Ffeat, BCE+FT', 'cnn_v12_ft_em2single_dual_fracbce/run_*', 9221,
              (17.45, 66.39, 90.54, 96.71, 97.93, 69.61, 0.29)),
 'e3v2f_ft': (r'E3+V2+\Ffeat, FT', 'cnn_v12_ft_em2stride_conv2x2_frac/run_*', 5421,
              (16.93, 64.31, 90.08, 97.48, 98.55, 69.08, 0.35)),
 'mlp_ctf_if': (r'MLP, \CT+\Ffeat, IF', 'nn_only_core_fraction_s*/run_c002_*', 2049,
              (16.61, 62.93, 89.00, 97.17, 98.53, 68.40, 0.21)),
 # OR analysis
 'mlp_ctf_if_or': (r'MLP, \CT+\Ffeat, IF, OR', 'power_law_extended_s*/run_c011_*', 2049,
              (12.75, 59.61, 88.83, 97.97, 99.69, 66.97, 0.44)),
 'mlp_ctf_p1_or': (r'MLP, \CT+\Ffeat, $p{=}-1$, OR', 'power_law_extended_s*/run_c010_*', 2049,
              (9.99, 60.11, 90.01, 98.03, 99.73, 66.66, 0.44)),
 'g002': (r'MLP, \CT+\Ffeat, FT, OR ($b{=}0.001$)', 'constrained_stage_g_or_s*/run_g002_*', 2049, None),
 'g003': (r'MLP, \CT+\Ffeat, FT, OR ($b{=}0.0015$)', 'constrained_stage_g_or_s*/run_g003_*', 2049, None),
 # progression (Section 5.2 b)
 'v4':      (r'V4 (\CT), IF', 'cnn_v6_dualbranch_raw_s*/run_*', 5941,
              (13.58, 59.78, 88.24, 96.80, 98.04, 66.68, 0.22)),
 'v4f':     (r'V4+\Ffeat, IF', 'cnn_v7_gpu0*/run_v7_dual_frac_*', 5973,
              (16.83, 62.57, 88.69, 96.83, 98.32, 68.28, 0.23)),
 'e1v4':    (r'E1+V4, IF', 'cnn_v9_em2single_dual/run_*', 9189,
              (17.11, 63.78, 89.63, 96.88, 97.65, 68.67, 0.28)),
 'e1v4f':   (r'E1+V4+\Ffeat, IF', 'cnn_v10_dual_em2_frac/run_*', 9221,
              (18.51, 64.73, 89.49, 96.43, 97.71, 69.15, 0.14)),
 'e1ctf':   (r'E1, flat \CT+\Ffeat, IF', 'cnn_v14_em2single_ctflat/run_*', 5297,
              (17.11, 64.05, 89.80, 97.21, 98.19, 68.88, 0.19)),
 'mlp_ct':  (r'MLP, \CT, IF', 'nn_only_benchmark_s*/run_c002_*', 2017,
              (13.19, 59.79, 89.25, 97.94, 99.02, 67.03, 0.36)),
 'e1v4f_bce': (r'E1+V4+\Ffeat, BCE', 'cnn_v11_gpu*/run_v11_em2single_dual_frac_bce_*', 9221,
              (19.05, 66.91, 89.49, 95.28, 96.64, 69.53, 0.23)),
 'mlp_ctf_p1': (r'MLP, \CT+\Ffeat, $p{=}-1$', 'nn_only_core_fraction_s*/run_c001_*', 2049,
              (12.31, 62.68, 90.44, 97.90, 99.33, 67.81, 0.35)),
 'b1':      (r'MLP, \CT+\Ffeat, $p{=}-1$+FT', 'constrained_stage_h11_20_s*/run_h101_*', 2049,
              (13.73, 63.84, 90.50, 97.52, 98.91, 68.34, 0.42)),
}

def runs(key):
    return sorted(glob.glob(B + M[key][1] + '/metrics.json'))

def stats(key):
    """Three-seed summary; refuse missing/duplicate evidence."""
    ps = runs(key)
    if len(ps) == 3:
        s = lib.summarize(ps)
        rs = s['rs']
        eff = np.mean([r['eff'] for r in rs], 0)
        err = np.sqrt(np.sum([r['err'] ** 2 for r in rs], 0)) / len(rs)
        base = np.mean([r['base'] for r in rs], 0)
        bsp = [r['m']['classifier_calibrations']['0.5%'].get('tob_fpr_budget') for r in rs]
        return dict(src='metrics', bands=s['bands'], glob=s['glob'], se=s['glob_se'], eff=eff, err=err,
                    base=base, base_bands=s['base_bands'], base_glob=s['base_glob'], fpr=s['fpr'],
                    base_fpr=[r['base_fpr'] for r in rs], seeds=s['seeds'], budgets=bsp, paths=ps)
    raise RuntimeError(f'{key}: expected 3 metric files, found {len(ps)}; restore metrics first')

BASE_BANDS = np.array([2.28, 35.85, 81.60, 96.89, 99.34]); BASE_GLOB = 57.37

def kparams(n):
    return f'{n/1000:.1f}K'

def row(label, st, params, bold=None):
    b = st['bands']
    cells = [f'{x:.2f}' for x in b]
    for k in range(5):  # mark bands where the model is below the baseline
        if b[k] < BASE_BANDS[k]:
            cells[k] = r'\textcolor{cbred}{' + cells[k] + '}'
    g = f"{st['glob']:.2f}$\\pm${st['se']:.2f}"
    tag = '' if st['src'] == 'metrics' else '  % REPORT fallback'
    return f"{label} & " + ' & '.join(cells) + f" & {g} & {params} \\\\{tag}"

# ----------------------------------------------------------------------------
# Main figure
# ----------------------------------------------------------------------------
COL = {'e2v4f_or': '#2e9e5b', 'v3f_or': '#c9a227', 'e1v4f_ft': '#2a78d6', 'e3v2f_ft': '#eb6834', 'mlp_ctf_if': '#8e5bd6'}
MAIN = ['e2v4f_or', 'e1v4f_ft', 'e3v2f_ft', 'mlp_ctf_if', 'v3f_or']
MAIN_TAG = {'e1v4f_ft': '(2)', 'e3v2f_ft': '(3)', 'mlp_ctf_if': '(4)', 'e2v4f_or': '(1)', 'v3f_or': '(5)'}
CRIT = {'e2v4f_or': r'(1) all bands', 'e1v4f_ft': r'(2) global', 'e3v2f_ft': r'(3) global, $<$6K', 'mlp_ctf_if': r'(4) global, $<$3K', 'v3f_or': r'(5) all bands, $<$3K'}

def fd(x, mid, slope, plat):
    return plat / (1 + np.exp(-slope * (x - mid)))

def main_figure():
    fig = plt.figure(figsize=(6.75, 3.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.45, 1], height_ratios=[2.3, 1], wspace=0.22, hspace=0.08,
                          left=0.07, right=0.985, top=0.885, bottom=0.11)
    ax = fig.add_subplot(gs[0, 0]); axd = fig.add_subplot(gs[1, 0], sharex=ax); axp = fig.add_subplot(gs[:, 1])
    ins = ax.inset_axes([0.50, 0.08, 0.47, 0.52])
    base = None
    for k in MAIN:
        st = stats(k)
        if st['eff'] is None:
            print('main figure: no binned data for', k); continue
        if base is None:
            base = st['base']
            for a in (ax, ins):
                a.plot(CENT, base, color='black', lw=1.1, label='eFEX baseline ($p_T^{\\mathrm{TOB}}\\geq C$)', zorder=5)
        c = COL[k]
        lab = f"{MAIN_TAG[k]} {M[k][0]}"
        lab = lab.replace(r'\Ffeat', 'F').replace(r'\CT', 'CT')
        for a in (ax, ins):
            a.plot(CENT, st['eff'], color=c, lw=0.9, marker='o', ms=1.6, label=f"{lab} ({kparams(M[k][2])})")
            a.fill_between(CENT, st['eff'] - st['err'], st['eff'] + st['err'], color=c, alpha=0.18, lw=0)
        d = (st['eff'] - st['base']) * 100
        de = st['err'] * 100
        axd.plot(CENT, d, color=c, lw=0.9, marker='o', ms=1.4)
        axd.fill_between(CENT, d - de, d + de, color=c, alpha=0.15, lw=0)
    axd.axhline(0, color='black', lw=0.8)
    ax.set_xlim(10, 120); ax.set_ylim(0, 1.02); ax.set_ylabel('Signal efficiency')
    ax.tick_params(labelbottom=False); ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, loc='lower left', bbox_to_anchor=(0.03, 1.0), ncol=3, fontsize=5.8, handlelength=1.6, borderaxespad=0.2, columnspacing=1.2)
    ins.set_xlim(60, 120); ins.set_ylim(0.93, 1.0); ins.tick_params(labelsize=5.5)
    ins.set_title('saturation region', fontsize=5.8, pad=2)
    for s in ins.spines.values(): s.set_linewidth(0.4)
    axd.set_xlabel('True $p_T$ [GeV]'); axd.set_ylabel('Gain [points]')
    axd.set_ylim(-6, 38); axd.spines[['top', 'right']].set_visible(False)
    for a in (ax, axd): a.grid(alpha=0.25, lw=0.3)
    # prior work, reported fits
    x = np.linspace(10, 120, 400)
    nn = fd(x, 26.8741, 0.1830, 0.9157); bl = fd(x, 33.0376, 0.2557, 1.0)
    axp.plot(x, bl, color='black', lw=1.1, label='tob_pt baseline (fit)')
    axp.plot(x, nn, color='#6b6b6b', lw=1.1, ls='--', label='prior best NN, CP+CT (fit, 2.1K)')
    i = np.where(np.diff(np.sign(nn - bl)))[0]
    xc = x[i[-1]] if len(i) else None
    if xc is not None:
        axp.axvline(xc, color='#b0b0b0', lw=0.5, ls=':')
        axp.annotate(f'curves cross\nat {xc:.0f} GeV', xy=(xc, fd(xc, 26.8741, 0.1830, 0.9157)), xytext=(52, 0.55),
                     fontsize=5.8, arrowprops=dict(arrowstyle='-', lw=0.4, color='#6b6b6b'))
    axp.axhline(0.9157, color='#6b6b6b', lw=0.4, ls=':')
    axp.text(119, 0.905, 'plateau 0.916', ha='right', va='top', fontsize=5.8, color='#52514e')
    axp.set_xlim(10, 120); axp.set_ylim(0, 1.02); axp.set_xlabel('True $p_T$ [GeV]')
    axp.set_title('Prior work, as reported in [5]', fontsize=6.5, pad=3)
    axp.legend(frameon=False, loc='lower right', fontsize=5.6, handlelength=1.8, bbox_to_anchor=(1.0, 0.1))
    axp.spines[['top', 'right']].set_visible(False); axp.grid(alpha=0.25, lw=0.3)
    ax.text(-0.075, 1.0, '(a)', transform=ax.transAxes, fontsize=7, fontweight='bold', va='bottom', ha='left')
    axp.text(-0.02, 1.0, '(b)', transform=axp.transAxes, fontsize=7, fontweight='bold', va='bottom', ha='right')
    fig.savefig(f'{PAPER}/figures/fig_results_main.pdf'); fig.savefig(f'{PAPER}/figures/fig_results_main.png', dpi=220)
    return xc

def main_table():
    lines = []
    for k in MAIN:
        st = stats(k)
        lines.append(CRIT[k] + ' & ' + row(M[k][0], st, kparams(M[k][2])))
    lines.append(r'\midrule')
    # Band means supplied by the previous researchers; global confirmed by the authors.
    # Do not infer a pointwise curve or global efficiency from these five means.
    lines.append(r'--- & Prior best NN, \CP+\CT~\cite{kernan2026} & 12.59 & 60.07 & 86.44 & 90.90 & 91.63 & 62.53 & 2.1K \\')
    lines.append(r'--- & eFEX baseline & 2.28 & 35.85 & 81.60 & 96.89 & 99.34 & 57.37 & --- \\')
    head = [r'\begin{tabular}{@{}llrrrrrrr@{}}', r'\toprule',
            r'Criterion & Model & 10--25 & 25--40 & 40--60 & 60--100 & 100--120 & Global & Params \\', r'\midrule']
    open(f'{PAPER}/tables/tab_main.tex', 'w').write('\n'.join(head + lines + [r'\bottomrule', r'\end{tabular}']) + '\n')

def write_tabular(name, lines, first='Model'):
    head = [r'\begin{tabular}{@{}lrrrrrrr@{}}', r'\toprule',
            first + r' & 10--25 & 25--40 & 40--60 & 60--100 & 100--120 & Global & Params \\', r'\midrule']
    tail = [r'\bottomrule', r'\end{tabular}']
    open(f'{PAPER}/tables/{name}', 'w').write('\n'.join(head + lines + tail) + '\n')

if __name__ == '__main__':
    what = sys.argv[1:] or ['main']
    if 'main' in what:
        xc = main_figure(); main_table(); print('prior crossing', xc)
    if 'dump' in what:
        for k in M:
            try:
                st = stats(k)
                print(f"{k:14s} {st['src']:7s}", np.round(st['bands'], 2), f"{st['glob']:.2f}+-{st['se']:.2f}",
                      st.get('budgets', ''), [round(f, 5) for f in st.get('fpr', [])])
            except Exception as e:
                print(k, 'ERR', e)

# ----------------------------------------------------------------------------
# Analysis figure: (a) cost and benefit of the OR after reweighting,
#                  (b) the CT-convolution -> EM2 -> Frac progression.
# ----------------------------------------------------------------------------
# chapter-table values of the project report (MLP era; these tables reproduce exactly)
OR_PAIRS = {  # input: (IF+OR bands, IF NN-only bands)
    r'CT':    ((11.25, 57.28, 88.23, 97.83, 99.37), (13.19, 59.79, 89.25, 97.94, 99.02)),
    r'CT+F':  ((12.75, 59.61, 88.83, 97.97, 99.69), (16.61, 62.93, 89.00, 97.17, 98.53)),
    r'SUM+F': ((12.31, 58.24, 88.21, 97.86, 99.61), (15.85, 60.29, 88.34, 97.32, 98.62)),
}
PROG = ['v4', 'v4f', 'e1v4', 'e1v4f', 'e1ctf']
PROG_COL = ['#b9c7d9', '#7fa6d6', '#eb9d6f', '#d65f2a', '#2e9e5b']
BANDLAB = ['10–25', '25–40', '40–60', '60–100', '100–120']

def analysis_figure():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.75, 1.75), gridspec_kw=dict(width_ratios=[1, 1.35]))
    x = np.arange(5)
    cols = ['#7fa6d6', '#2a78d6', '#8e5bd6']
    w = 0.26
    for j, (k, (orb, nnb)) in enumerate(OR_PAIRS.items()):
        d = np.array(orb) - np.array(nnb)
        a1.bar(x + (j - 1) * w, d, w * 0.95, color=cols[j], label=k)
    a1.axhline(0, color='black', lw=0.6)
    a1.set_xticks(x, BANDLAB); a1.set_ylabel('OR $-$ NN-only [points]')
    a1.set_title('(a) What the OR adds after reweighting (IF)', fontsize=6.8, pad=3)
    a1.legend(frameon=False, fontsize=5.8, loc='lower right', handlelength=1.0)
    a1.spines[['top', 'right']].set_visible(False); a1.grid(axis='y', alpha=0.25, lw=0.3)
    w = 0.16
    for j, k in enumerate(PROG):
        st = stats(k)
        d = st['bands'] - BASE_BANDS
        lab = M[k][0].replace(r'\Ffeat', 'F').replace(r'\CT', 'CT')
        a2.bar(x + (j - 2) * w, d, w * 0.95, color=PROG_COL[j], label=f"{lab} ({st['glob']:.1f})")
    a2.axhline(0, color='black', lw=0.6)
    a2.set_xticks(x, BANDLAB); a2.set_ylabel('Gain over baseline [points]')
    a2.set_title('(b) From the core-tensor convolution to EM2 and Frac', fontsize=6.8, pad=3)
    a2.legend(frameon=False, fontsize=5.6, loc='upper right', handlelength=1.0, title='model (global %)', title_fontsize=5.6)
    a2.spines[['top', 'right']].set_visible(False); a2.grid(axis='y', alpha=0.25, lw=0.3)
    for a in (a1, a2):
        a.tick_params(labelsize=6); a.set_xlabel('True $p_T$ band [GeV]', fontsize=6.5)
    fig.tight_layout(pad=0.3, w_pad=1.0)
    fig.savefig(f'{PAPER}/figures/fig_results_analysis.pdf'); fig.savefig(f'{PAPER}/figures/fig_results_analysis.png', dpi=220)

if __name__ == '__main__' and 'analysis' in sys.argv[1:]:
    analysis_figure()

def ft_table():
    lines = []
    for a, b in [('mlp_ctf_p1', 'b1'), ('e1v4f_bce', 'e1v4f_ft')]:
        for k in (a, b):
            lines.append(row(M[k][0], stats(k), kparams(M[k][2])))
        if a == 'mlp_ctf_p1': lines.append(r'\addlinespace[2pt]')
    write_tabular('tab_ft.tex', lines)

if __name__ == '__main__' and 'ft' in sys.argv[1:]:
    ft_table()
