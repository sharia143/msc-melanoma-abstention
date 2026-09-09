"""
Concept inconsistency (rough-set profiles) on Derm7pt and MILK10k, and
concept-conflict abstention. Dissertation Sections 3.3-3.4 and 4.2-4.3.

Sharia Alam, MSc Data Science, Manchester Metropolitan University, 2026.
Needs Derm7pt meta.csv and the MILK10k metadata / ground-truth CSVs.
"""
import numpy as np, pandas as pd

D7_META = 'Derm7pt/meta/meta.csv'
MILK = 'MILK10k'

# ---------------------------------------------------------------- rough-set analysis
def rough_set(signatures, y):
    """Profiles (identical concept signatures), the inconsistent ones, boundary region,
    dependency degree gamma and the accuracy ceiling of any concept-only classifier."""
    g = pd.DataFrame(dict(s=signatures, y=y)).groupby('s')['y'].agg(size='size', pos='sum')
    g['neg'] = g['size'] - g['pos']
    g['incons'] = (g.pos > 0) & (g.neg > 0)
    U = len(y)
    POS = int(g.loc[~g.incons, 'size'].sum())                       # consistent images
    MAJ = int(g.loc[g.incons, ['pos', 'neg']].max(axis=1).sum())    # majority side of the rest
    in_boundary = pd.Series(signatures).isin(g[g.incons].index).values
    g['cr'] = g[['pos', 'neg']].min(axis=1) / g['size']             # conflict ratio
    stats = dict(profiles=len(g), inconsistent=int(g.incons.sum()),
                 boundary_images=int(in_boundary.sum()), boundary_pct=100 * in_boundary.mean(),
                 melanoma_in_boundary=int(y[in_boundary].sum()),
                 melanoma_boundary_pct=100 * y[in_boundary].sum() / max(y.sum(), 1),
                 gamma=POS / U, ceiling=(POS + MAJ) / U)
    return stats, g, in_boundary

# Derm7pt: seven clinician-assigned criteria, discrete values
m = pd.read_csv(D7_META)
CON7 = ['pigment_network', 'streaks', 'pigmentation', 'regression_structures',
        'dots_and_globules', 'blue_whitish_veil', 'vascular_structures']
y7 = m.diagnosis.str.startswith('melanoma').astype(int).values
sig7 = m[CON7].astype(str).agg('|'.join, axis=1).values
stats7, g7, bnd7 = rough_set(sig7, y7)
print('Derm7pt', {k: round(v, 3) if isinstance(v, float) else v for k, v in stats7.items()})
# published values (Napoles et al. 2026): 305 profiles, 50 inconsistent, 306 boundary images,
# 136 of 252 melanomas in the boundary, gamma 0.697, ceiling 0.921

# MILK10k: seven MONET concept probabilities, thresholded at 0.5
meta = pd.read_csv(f'{MILK}/MILK10k_Training_Metadata.csv')
gt = pd.read_csv(f'{MILK}/MILK10k_Training_GroundTruth.csv')
CONM = [c for c in meta.columns if c.startswith('MONET_')]
d = meta[meta.image_type == 'dermoscopic'].merge(gt[['lesion_id', 'MEL']], on='lesion_id')
ym = d.MEL.astype(int).values
Xm = d[CONM].values.astype(float)
sigm = np.array([''.join(r) for r in (Xm > 0.5).astype(int).astype(str)])
statsm, gm, bndm = rough_set(sigm, ym)
print('MILK10k', {k: round(v, 3) if isinstance(v, float) else v for k, v in statsm.items()},
      '| trivial baseline', round(1 - ym.mean(), 3))

for thr in (0.4, 0.5, 0.6):                                  # threshold sensitivity
    s_, *_ = rough_set(np.array([''.join(r) for r in (Xm > thr).astype(int).astype(str)]), ym)
    print(f'  threshold {thr}: boundary {s_["boundary_pct"]:.1f}%  ceiling {s_["ceiling"]:.3f}')

# ---------------------------------------------------------------- concept-conflict abstention
def cca(X, y, tau, loo=True):
    """Kernel local melanoma rate. Returns (kappa, prediction, rate); kappa = min(p, 1-p).
    With the lesion itself included (loo=False) the score tends to the rough-set conflict
    ratio as tau -> 0; evaluation uses leave-one-out so that a lesion never votes for itself."""
    X = np.asarray(X, float)
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    if loo:
        np.fill_diagonal(d2, np.inf)
    W = np.exp(-d2 / (2 * tau ** 2))
    p = (W * y[None, :]).sum(1) / np.maximum(W.sum(1), 1e-12)
    return np.minimum(p, 1 - p), (p >= 0.5).astype(int), p

def selective_report(kappa, pred, y, coverages=(1.0, 0.9, 0.8, 0.7, 0.6)):
    """Refuse the most conflicted cases first; risk, melanoma retained, recall, precision."""
    order = np.argsort(kappa, kind='stable'); rows = []
    for c in coverages:
        idx = order[:int(c * len(y))]; yr, pr = y[idx], pred[idx]
        rows.append(dict(coverage=c, risk=1 - (pr == yr).mean(), melanoma_retained=yr.sum() / y.sum(),
                         recall=(pr[yr == 1] == 1).mean() if yr.sum() else np.nan,
                         precision=(yr[pr == 1] == 1).mean() if pr.sum() else np.nan))
    return pd.DataFrame(rows)

# reduction property: with the lesion included, tau -> 0 reproduces the rough-set conflict ratio
X7 = pd.get_dummies(m[CON7].astype(str)).values.astype(float)
k_hard = pd.Series(sig7).map(g7['cr']).values
k0, pred0, _ = cca(X7, y7, tau=1e-3, loo=False)
assert np.allclose(k0, k_hard) and abs((pred0 == y7).mean() - stats7['ceiling']) < 1e-9

pred_hard = pd.Series(sig7).map(g7[['pos', 'neg']].idxmax(axis=1).map({'pos': 1, 'neg': 0})).values
print('\nDerm7pt, concept-conflict abstention'); print(selective_report(k_hard, pred_hard, y7).round(3))

Xm_s = (Xm - Xm.mean(0)) / Xm.std(0)
km, predm, _ = cca(Xm_s, ym, tau=0.75)
print('\nMILK10k, concept-conflict abstention'); print(selective_report(km, predm, ym).round(3))
# recall stays near zero at every coverage: the MONET vocabulary does not separate melanoma,
# so the rule defers melanoma instead of detecting it (a negative result for the concept set)
