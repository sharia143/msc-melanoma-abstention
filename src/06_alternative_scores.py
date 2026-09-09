"""
Twelve alternative abstention scores on the same frozen features, the thesis
thresholding rule applied to each, leave-one-class-out with a linear probe, and
conformal risk control on the melanoma miss rate. Dissertation Sections 4.9-4.10
(Tables 4.13-4.14, Figures 4.22-4.25).

Sharia Alam, MSc Data Science, Manchester Metropolitan University, 2026.
Uses the arrays produced by 05_abstaining_cascade.py (F_*, P, NOV, CONF, Y, REF, W2, B2).
"""
import json
import numpy as np, pandas as pd
import ood_scores as oods

K, GATE1_PCT, GATE2_PCT, SEED = 10, 95, 80, 42
A = np.load('bundles/arrays.npz')                                    # written by 05_abstaining_cascade.py
F_TRAIN, F_CAL, F_TEST, Y, Y_CAL = A['F_TRAIN'], A['F_CAL'], A['F_TEST'], A['Y'], A['Y_CAL']
P, NOV, CONF, P_CAL, NOV_CAL, CONF_CAL = (A[k] for k in ('P', 'NOV', 'CONF', 'P_CAL', 'NOV_CAL', 'CONF_CAL'))
P_DFU, REF, W2, B2 = A['P_DFU'], A['REF'], A['W2'], A['B2']
FEATS = {k[4:]: A[k] for k in A.files if k.startswith('POP_')}
cfg = json.load(open('bundles/C4_seed42/config.json'))
TAU, FLOOR, NEAR = cfg['decision_threshold'], cfg['thresholds']['confidence_floor'], cfg['thresholds']['near_ood_knn']
split = pd.read_csv('bundles/split.csv')
dx_train, dx_test = split[split.part == 'train'].dx.values, split[split.part == 'test'].dx.values
logits_of = lambda F: F @ W2.T + B2

def cascade(nov, conf, p, far, near, floor=FLOOR, tau=TAU):
    gate = np.full(len(nov), '', dtype=object)
    gate[nov > far] = 'far_ood'; gate[(nov <= far) & (nov > near)] = 'near_ood'
    gate[(nov <= near) & (conf < floor)] = 'low_confidence'
    return gate, gate == '', np.where(p >= tau, 'melanoma', 'not melanoma')

maha2 = oods.Mahalanobis().fit(F_TRAIN, split[split.part == 'train'].mel.values)
maha11 = oods.Mahalanobis().fit(F_TRAIN, dx_train)
vim2 = oods.ViM().fit(F_TRAIN, W2, B2)
probe = oods.LinearProbe(C=0.5, seed=SEED).fit(F_TRAIN, dx_train)   # eleven-class head on the binary model's features
vim11 = probe.vim(F_TRAIN)
print(f'probe accuracy {probe.accuracy(F_TEST, dx_test):.3f} (majority class {pd.Series(dx_test).value_counts(normalize=True).iloc[0]:.3f})')

knn_cal, e11_cal = oods.knn_mean(F_CAL, REF, K), oods.energy(probe.logits(F_CAL))
SCORES = {   # every score is "higher = more novel"
    'kNN mean (deployed)': lambda F: oods.knn_mean(F, REF, K),
    'kNN k-th': lambda F: oods.knn_kth(F, REF, K),
    'Mahalanobis, 2 classes': maha2.score,
    'Mahalanobis, 11 classes': maha11.score,
    'relative Mahalanobis, 11 classes': maha11.rmd,
    'MSP, binary head': lambda F: oods.msp(logits_of(F)),
    'max logit, binary head': lambda F: oods.max_logit(logits_of(F)),
    'energy, binary head': lambda F: oods.energy(logits_of(F)),
    'ViM, binary head': vim2.score,
    'MSP, 11-class probe': lambda F: oods.msp(probe.logits(F)),
    'energy, 11-class probe': lambda F: oods.energy(probe.logits(F)),
    'ViM, 11-class probe': vim11.score,
    'kNN + energy-11, rank fusion': lambda F: oods.rank_fusion(oods.knn_mean(F, REF, K), oods.energy(probe.logits(F)), knn_cal, e11_cal),
}

rows = []
for name, fn in SCORES.items():
    s_cal, s_test = fn(F_CAL), fn(F_TEST)
    t95, t80 = np.percentile(s_cal, GATE1_PCT), np.percentile(s_cal, GATE2_PCT)   # the thesis rule, same for every score
    row = dict(score=name, test_refused=(s_test > t80).mean(), test_melanomas_refused=int(((s_test > t80) & (Y == 1)).sum()))
    for pop, F in FEATS.items():
        s_pop = fn(F)
        row[f'auroc: {pop}'] = oods.auroc(s_test, s_pop)
        row[f'refused: {pop}'] = (s_pop > t80).mean()
    F_dfu = FEATS['DFUC2021, foot ulcers']
    _, a, c = cascade(fn(F_dfu), np.maximum(P_DFU, 1 - P_DFU), P_DFU, far=t95, near=t80)
    row['ulcers called melanoma'] = int((a & (c == 'melanoma')).sum())
    rows.append(row)
print(pd.DataFrame(rows).set_index('score').round(3))

# leave-one-class-out with the probe (the backbone has seen every class; this tests the head)
HOLD = [c for c in ('BKL', 'SCCKA', 'AKIEC', 'NV') if (dx_test == c).sum() >= 20]
print(pd.DataFrame(oods.loco_probe(F_TRAIN, dx_train, F_TEST, dx_test, HOLD, knn_ref_unit=REF, k=K)).set_index('held_out').round(3))
# the binary backbone retrained without each class gave a mean AUROC of 0.598 (Full_Pipeline_v3)

# conformal risk control on the melanoma miss rate, defined within the cascade
mel_cal = Y_CAL == 1
floors, cuts = np.linspace(0.5, 1.0, 501), np.round(np.linspace(0.90, 0.01, 890), 4)
pass_cal = NOV_CAL[mel_cal] <= NEAR                                  # a melanoma the novelty gates refuse cannot be missed
Lf = oods.miss_losses_floor(P_CAL[mel_cal], CONF_CAL[mel_cal], floors, TAU, keep=pass_cal)
Lc = oods.miss_losses_cut(P_CAL[mel_cal], cuts, keep=pass_cal & (CONF_CAL[mel_cal] >= FLOOR))
for alpha in (0.10, 0.15, 0.20):
    jf, floor_hat = oods.crc_threshold(Lf, floors, alpha)
    jc, cut_hat = oods.crc_threshold(Lc, cuts, alpha)
    print(f'alpha {alpha:.2f}: floor {floor_hat}, cut {cut_hat}  (thesis: floor {FLOOR:.4f}, cut {TAU:.3f})')
