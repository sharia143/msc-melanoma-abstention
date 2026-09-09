"""
The extension experiments: three seeds per configuration, bootstrap intervals, the
skin-tone and sex audit, the wound-aware stage with its pets control, sensitivity of
the novelty gate to k and to the reference set, an eleven-class backbone under
leave-one-class-out, the conflict rule on the five readable concepts, and the
before/after summary. Dissertation Sections 3.7, 4.3, 4.5, 4.7, 4.9 and 4.11-4.13.

Sharia Alam, MSc Data Science, Manchester Metropolitan University, 2026.
Uses bundles/arrays.npz (05), bundles/split.csv (04) and the C1-C4 bundles (04).
Every number it produces was written to v5_numbers.tex and read into the dissertation.
"""
import json
import numpy as np, pandas as pd
from scipy import stats as sstats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
import ood_scores as oods

SEED, K, GATE1_PCT, GATE2_PCT = 42, 10, 95, 80
SEEDS, CONFIGS = (42, 7, 2026), ('C1', 'C2', 'C3', 'C4')
A = np.load('bundles/arrays.npz')
F_TRAIN, F_CAL, F_TEST, Y, Y_CAL = A['F_TRAIN'], A['F_CAL'], A['F_TEST'], A['Y'], A['Y_CAL']
P, NOV, CONF, P_CAL, NOV_CAL, CONF_CAL = (A[k] for k in ('P', 'NOV', 'CONF', 'P_CAL', 'NOV_CAL', 'CONF_CAL'))
P_DFU, NOV_DFU, GATE_DFU, REF = A['P_DFU'], A['NOV_DFU'], A['GATE_DFU'], A['REF']
FEATS = {k[4:]: A[k] for k in A.files if k.startswith('POP_')}
F_DFU, F_PETS = FEATS['DFUC2021, foot ulcers'], FEATS['Oxford-IIIT pets']
cfg = json.load(open('bundles/C4_seed42/config.json'))
TAU, FLOOR = cfg['decision_threshold'], cfg['thresholds']['confidence_floor']
FAR, NEAR = cfg['thresholds']['far_ood_knn'], cfg['thresholds']['near_ood_knn']
split = pd.read_csv('bundles/split.csv')
train, cal, test = (split[split.part == p_].reset_index(drop=True) for p_ in ('train', 'calibration', 'test'))

def cascade(nov, conf, p, far=FAR, near=NEAR, floor=FLOOR, tau=TAU):
    gate = np.full(len(nov), '', dtype=object)
    gate[nov > far] = 'far_ood'; gate[(nov <= far) & (nov > near)] = 'near_ood'
    gate[(nov <= near) & (conf < floor)] = 'low_confidence'
    return gate, gate == '', np.where(p >= tau, 'melanoma', 'not melanoma')

GATE, ANS, CALL = cascade(NOV, CONF, P)
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- A. seeds: the four configurations, three runs each (Table 4.10)
# each bundle was produced by 04_train_classifier.py under one protocol; evaluated here at its own thresholds
seeds = pd.DataFrame([json.load(open(f'bundles/{c}_seed{s}/eval.json')) for c in CONFIGS for s in SEEDS])
cols = ['auroc', 'ece', 'sensitivity_answered', 'specificity_answered', 'coverage', 'melanomas_answered', 'tau', 'T']
print(seeds.groupby('config')[cols].agg(['mean', 'std']).round(3))
for c in CONFIGS[1:]:                                                # paired differences against C1, seed by seed
    d = seeds[seeds.config == c].set_index('seed')[cols] - seeds[seeds.config == 'C1'].set_index('seed')[cols]
    print(c, 'minus C1:', d[['sensitivity_answered', 'melanomas_answered', 'ece']].mean().round(3).to_dict())
# temperature scaling is monotone, so C3 and C4 make identical decisions; only ECE and the cut differ

# ---------------------------------------------------------------- B. bootstrap intervals on the deployed model (2,000 resamples)
def boot_ci(fn, n, B=2000):
    vals = [fn(rng.integers(0, n, n)) for _ in range(B)]
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)
n = len(Y)
for name, fn in {'AUROC': lambda i: roc_auc_score(Y[i], P[i]) if len(set(Y[i])) > 1 else np.nan,
                 'coverage': lambda i: ANS[i].mean(),
                 'sensitivity answered': lambda i: (CALL[i][ANS[i] & (Y[i] == 1)] == 'melanoma').mean(),
                 'melanomas answered': lambda i: (ANS[i] & (Y[i] == 1)).sum()}.items():
    print(f'{name:22s} 95% CI {boot_ci(fn, n)}')

# ---------------------------------------------------------------- C. who is refused: skin tone and sex (Table 4.15)
# MILK10k skin_tone_class runs from 0 (very dark) to 5 (very light); 0-2 pooled because they are rare
tone = test.skin_tone_class.map(lambda v: None if pd.isna(v) else ('0-2' if v <= 2 else str(int(v)))).values
def audit(groups, ans, y, call, label):
    for g, m in groups:
        if not m.any():
            continue
        refused = (~ans[m]).mean()
        lo, hi = boot_ci(lambda i: (~ans[m][i]).mean(), m.sum(), B=800)
        mel, ben = ans & m & (y == 1), ans & m & (y == 0)
        print(f'{label:32s} {g:6s} n {m.sum():4d} (mel {int((m & (y == 1)).sum()):3d})  refused {100 * refused:5.1f}% '
              f'[{100 * lo:.1f}, {100 * hi:.1f}]  sens. answered {(call[mel] == "melanoma").mean() if mel.any() else np.nan:.3f} '
              f'false alarms {(call[ben] == "melanoma").mean() if ben.any() else np.nan:.3f}')
audit([(g, tone == g) for g in ('0-2', '3', '4', '5')], ANS, Y, CALL, 'dermoscopic test split')
ct = pd.crosstab(tone, ~ANS)
chi2, pval, dof, _ = sstats.chi2_contingency(ct.values)
print(f'refusal x tone: chi2 {chi2:.2f}, dof {dof}, p {pval:.3f}')
sex = test.sex.fillna('unknown').str.lower().values
audit([(s, sex == s) for s in ('female', 'male')], ANS, Y, CALL, 'dermoscopic test split, by sex')
# the same audit is run on the clinical close-ups of the same lesions (features in FEATS)

# ---------------------------------------------------------------- E. a wound-aware stage in front of the cascade (Section 4.12)
WOUND_FRAC = 0.20
fit_idx = rng.choice(len(F_DFU), int(WOUND_FRAC * len(F_DFU)), replace=False)   # stratified by DFUC2021 label in the run
hold = np.setdiff1d(np.arange(len(F_DFU)), fit_idx)

def fit_stage(F_pos, F_neg, seed=SEED):
    X = np.vstack([F_neg, F_pos]); y = np.r_[np.zeros(len(F_neg)), np.ones(len(F_pos))]
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight='balanced', max_iter=3000, random_state=seed)).fit(X, y)

stage = fit_stage(F_DFU[fit_idx], F_TRAIN)
thr = np.percentile(stage.predict_proba(F_CAL)[:, 1], 99)          # refuses at most 1% of calibration lesions
s_test, s_hold = stage.predict_proba(F_TEST)[:, 1], stage.predict_proba(F_DFU[hold])[:, 1]
print(f'wound stage AUROC {oods.auroc(s_test, s_hold):.4f}; refuses {100 * (s_test > thr).mean():.1f}% of test lesions, '
      f'{100 * (s_hold > thr).mean():.1f}% of held-out ulcers')
for name, F in FEATS.items():
    print(f'  refused by the stage: {name:34s} {100 * (stage.predict_proba(F)[:, 1] > thr).mean():5.1f}%')
answered_dfu = GATE_DFU == ''
called = answered_dfu & (P_DFU >= TAU)
h_ans = answered_dfu[hold] & ~(s_hold > thr)
scale = len(F_DFU) / len(hold)
print(f'ulcers called melanoma: {called.sum()} -> {round((h_ans & (P_DFU[hold] >= TAU)).sum() * scale)} (held-out, scaled to all)')
ANS0 = ANS & ~(s_test > thr)
print(f'test coverage {100 * ANS.mean():.1f}% -> {100 * ANS0.mean():.1f}%; melanomas answered '
      f'{(ANS & (Y == 1)).sum()} -> {(ANS0 & (Y == 1)).sum()}')
# control: a stage trained on pet photographs instead of ulcers
ip = rng.permutation(len(F_PETS)); half = len(ip) // 2
stage_p = fit_stage(F_PETS[ip[:half]], F_TRAIN)
thr_p = np.percentile(stage_p.predict_proba(F_CAL)[:, 1], 99)
print(f'pets control: refuses {100 * (stage_p.predict_proba(F_PETS[ip[half:]])[:, 1] > thr_p).mean():.0f}% of pets but only '
      f'{100 * (stage_p.predict_proba(F_DFU[hold])[:, 1] > thr_p).mean():.0f}% of ulcers')

# ---------------------------------------------------------------- F. k and the reference set (Figure 4.17)
mel_tr = train.mel.values == 1
REFS = {'balanced 630 (deployed)': REF, 'all training lesions': oods.unit(F_TRAIN),
        'melanoma only': oods.unit(F_TRAIN[mel_tr]), 'benign only': oods.unit(F_TRAIN[~mel_tr])}
for ref_name, R in REFS.items():
    for k in (1, 5, 10, 20, 50):
        s_cal, s_te = oods.knn_mean(F_CAL, R, k), oods.knn_mean(F_TEST, R, k)
        t80 = np.percentile(s_cal, GATE2_PCT)
        print(f'{ref_name:26s} k {k:2d}  melanomas refused {int(((s_te > t80) & (Y == 1)).sum()):2d}  '
              f'AUROC ulcers {oods.auroc(s_te, oods.knn_mean(F_DFU, R, k)):.3f}  pets {oods.auroc(s_te, oods.knn_mean(F_PETS, R, k)):.3f}')
# whichever class is under-represented in the reference is refused more

# ---------------------------------------------------------------- G. an eleven-class backbone under leave-one-class-out (Figure 4.24)
# the backbone retrained with an eleven-class head, once on all classes and once without each of
# BKL, SCCKA, AKIEC and NV; the retrained model's scores separate the held-out type's test lesions from the rest
for tag in ('all', 'BKL', 'SCCKA', 'AKIEC', 'NV'):                   # written by 08_eleven_class_backbone.py
    r = json.load(open(f'bundles/multi_{tag}_seed{SEED}/eval.json'))
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k in
           ('held_out', 'test_macro_recall', 'melanoma_auroc_from_11class', 'auroc_energy', 'auroc_msp', 'auroc_knn')})

# ---------------------------------------------------------------- H. the conflict rule on the five concepts MONET reads above chance
def cca_table(Z, y, tau=1.0, coverages=(100, 90, 80, 70)):
    Z = (Z - Z.mean(0)) / np.maximum(Z.std(0), 1e-9)
    D = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1); Kk = np.exp(-D / (2 * tau ** 2)); np.fill_diagonal(Kk, 0.0)
    pk = (Kk @ y) / np.maximum(Kk.sum(1), 1e-12); kappa = np.minimum(pk, 1 - pk); pred = (pk >= 0.5).astype(int)
    order = np.argsort(kappa); out = []
    for cov in coverages:
        keep = order[:int(round(len(y) * cov / 100))]; yk, pk_ = y[keep], pred[keep]
        out.append(dict(coverage=cov, risk=(pk_ != yk).mean(), melanoma_retained=yk.sum() / y.sum(),
                        recall=((pk_ == 1) & (yk == 1)).sum() / max(yk.sum(), 1)))
    return pd.DataFrame(out)
d7 = pd.read_csv('Derm7pt/derm7pt_expert_and_monet_concepts.csv')      # seven expert criteria and seven MONET scores per case
y7 = d7.is_mel.values
EXPERT, MONET = [c for c in d7.columns if c.endswith('_expert')], [c for c in d7.columns if c.endswith('_monet')]
readable = [i for i, c in enumerate(EXPERT) if not c.startswith(('atypical_pigment_network', 'atypical_vascular'))]
for name, Z in (('expert, 7', d7[EXPERT].values), ('MONET, 7', d7[MONET].values),
                ('MONET, 5 readable', d7[MONET].values[:, readable]), ('expert, same 5', d7[EXPERT].values[:, readable])):
    print(name); print(cca_table(Z.astype(float), y7).round(3))

# ---------------------------------------------------------------- I. before and after (Table 4.18)
def ece(p, y, bins=10):
    pred, conf = (p >= 0.5).astype(int), np.maximum(p, 1 - p)
    edges = np.linspace(0.5, 1.0, bins + 1); e = 0.0
    for b in range(bins):
        m = (conf >= edges[b]) & (conf < edges[b + 1] if b < bins - 1 else conf <= edges[b + 1])
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)
pass_cal = NOV_CAL[Y_CAL == 1] <= NEAR
cuts = np.round(np.linspace(0.90, 0.01, 890), 4)
_, cut_crc = oods.crc_threshold(oods.miss_losses_cut(P_CAL[Y_CAL == 1], cuts, keep=pass_cal & (CONF_CAL[Y_CAL == 1] >= FLOOR)), cuts, 0.15)
before_after = [
    ('ISIC validation accuracy, released -> patient-aware split', 0.827, 0.772),          # 01_isic_patient_split.py
    ('ECE, before -> after temperature', ece(A['P_RAW'], Y), ece(P, Y)),
    ('test risk, all cases -> answered cases', ((P >= TAU) != Y).mean(), ((P[ANS] >= TAU) != Y[ANS]).mean()),
    ('coverage, no gates -> cascade (%)', 100.0, 100 * ANS.mean()),
    ('ulcers called melanoma, deployed -> with wound stage (scaled)', int(called.sum()), round((h_ans & (P_DFU[hold] >= TAU)).sum() * scale)),
    ('decision cut, hand rule -> conformal (alpha 0.15)', TAU, cut_crc),
]
print(pd.DataFrame(before_after, columns=['quantity', 'before', 'after']).round(3))
