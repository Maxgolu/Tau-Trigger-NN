"""Scan every staged run, group seeds, name the model and count its parameters."""
import glob, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib

B = lib.BASE
FEAT_DIM = {'core_tensors': 45, 'em2_best_3x3_fraction': 1, 'core_physics': 4, 'em2_all_cells': 144}

def feat_dim(name):
    if name in FEAT_DIM: return FEAT_DIM[name]
    if 'summar' in name or name.startswith('layer_'): return 15
    if re.search(r'_(sparsity|sum|dominance)$', name): return 1
    return None

def conv_out(h, k, s): return (h - k) // s + 1

def branch_params(br):
    c, h, w = br['shape']; p = 0
    for L in br['layers']:
        t = L.get('type', 'conv')
        if t == 'conv':
            k = L['kernel']; s = L.get('stride', 1); o = L['out_channels']
            p += k * k * c * o + o + 2 * o   # weights, bias, BatchNorm
            c, h, w = o, conv_out(h, k, s), conv_out(w, k, s)
        elif t in ('maxpool', 'pool', 'max_pool'):
            k = L.get('kernel', 2); s = L.get('stride', k)
            h, w = conv_out(h, k, s), conv_out(w, k, s)
        else:
            raise ValueError(t)
    return p, c * h * w

def classify_branch(br):
    ks = [(L.get('type', 'conv'), L.get('kernel'), L.get('stride', 1)) for L in br['layers']]
    if br['feature'] == 'core_tensors':
        conv = [k for t, k, s in ks if t == 'conv']
        return {(1,): 'V1', (2,): 'V2', (1, 2): 'V3'}.get(tuple(conv), 'V?' + str(conv))
    if br['feature'] == 'em2_all_cells':
        convs = [(k, s) for t, k, s in ks if t == 'conv']
        pools = [t for t, k, s in ks if t != 'conv']
        if len(convs) == 1: return 'E1'
        if convs[0][1] == 2: return 'E3'
        return 'E2'
    return '?'

def describe(cfg):
    feats = cfg.get('features_to_use', [])
    model = cfg.get('model') or {}
    head = model.get('head') or cfg.get('hidden_layers') or [32, 16]
    used = set(); conv_p = 0; conv_out_dim = 0; parts = []
    if model.get('name') == 'tensor_cnn':
        vs = []; es = []
        for br in model['branches']:
            p, o = branch_params(br); conv_p += p; conv_out_dim += o; used.add(br['feature'])
            b = classify_branch(br); (vs if b.startswith('V') else es).append(b)
        if sorted(vs) == ['V1', 'V2']: vs = ['V4']
        parts = es + vs
        for x in model.get('extra_scalar_features', []) or []: pass
        if model.get('include_raw') or any(br.get('include_raw') for br in model['branches']):
            used.discard('core_tensors')
    scal = 0; tags = []
    for f in feats:
        if f in used: continue
        d = feat_dim(f)
        if d is None: return None
        scal += d
        t = {'core_tensors': 'CT', 'em2_best_3x3_fraction': 'F', 'core_physics': 'CP'}.get(f, 'SUM' if d in (1, 15) and f not in ('em2_best_3x3_fraction',) else f)
        if t == 'SUM' and 'SUM' in tags: continue
        tags.append(t)
    dims = [conv_out_dim + scal] + list(head)
    head_p = sum(a * b + b for a, b in zip(dims[:-1], dims[1:])) + dims[-1] + 1
    params = conv_p + head_p
    arch = '+'.join(parts) if parts else 'MLP'
    if parts and 'CT' in tags and not any(p.startswith('V') for p in parts):
        tags = ['flat CT' if t == 'CT' else t for t in tags]
    # objective
    loss = cfg.get('loss', {'name': 'bce'}); ln = loss.get('name', 'bce')
    if ln == 'bce': obj = 'BCE'
    elif ln == 'energy_weighted_bce':
        wt = loss.get('weighting', {}); pr = wt.get('profile')
        obj = {'inverse_frequency': 'IF', 'alpha': f"a={wt.get('alpha')}", 'power_law': f"p={wt.get('power', wt.get('p'))}"}.get(pr, str(pr))
    else: obj = ln
    if ln == 'constrained_trigger':
        wp = (cfg.get('initialization') or {}).get('weights_path', '').lower()
        init = wp.split('/')[-2] if '/' in wp else wp
        obj = 'BCE+FT' if 'bce' in init else ('p=-1+FT' if 'nn_only_core_fraction' in wp and 'c001' in init else 'FT')
    clf = 'OR' if cfg.get('classifier', {}).get('name') == 'tob_nn_or' else 'NN'
    return dict(arch=arch, scalars='+'.join(tags), obj=obj, clf=clf, params=params)

def group_key(path):
    d = os.path.dirname(path); rel = os.path.relpath(d, B).replace('\\','/')
    rel = re.sub(r'_gpu\d\d', '_gpu', rel)
    return re.sub(r'_s(42|123|456)(?=[_/])', '', rel).rsplit('_2026', 1)[0]

def scan():
    ps = sorted(glob.glob(B + '**/metrics.json', recursive=True))
    groups = {}
    for p in ps:
        groups.setdefault(group_key(p), []).append(p)
    out = []
    for g, pp in sorted(groups.items()):
        seeds = sorted({lib.seed_of(p) for p in pp})
        if len(pp) != 3 or seeds != [42,123,456]: continue
        cfgp = [os.path.join(os.path.dirname(p), 'config.json') for p in pp]
        desc = None
        ex = [c for c in cfgp if os.path.exists(c)]
        if ex:
            desc = describe(json.load(open(ex[0])))
        s = lib.summarize(pp)
        out.append(dict(group=g, n=len(pp), seeds=seeds, desc=desc, bands=s['bands'], glob=s['glob'], se=s['glob_se'], paths=pp,
                        fpr=s['fpr'], base_bands=s['base_bands']))
    return out

if __name__ == '__main__':
    for r in scan():
        d = r['desc'] or {}
        print(f"{r['group'][:60]:60s} n={r['n']} {d.get('arch','?'):8s} {d.get('scalars',''):10s} {d.get('obj',''):8s} {d.get('clf',''):3s} {d.get('params','?'):>6} ",
              np.round(r['bands'], 2), f"{r['glob']:.2f}")
