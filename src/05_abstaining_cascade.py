"""
The deployed cascade on the test split and on the outside populations: features,
novelty, the three active gates, coverage and sensitivity among answered cases,
calibration, risk-coverage, what each gate contributes, refusal by population, and the
diabetic-foot-ulcer trial. Dissertation Sections 4.5-4.8 (Tables 4.11-4.12, Figures 4.7-4.21).

Sharia Alam, MSc Data Science, Manchester Metropolitan University, 2026.
Expects the bundle written by 04_train_classifier.py (C4, seed 42) and the split it used.
"""
import json, glob
import numpy as np, pandas as pd, torch, timm
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score, confusion_matrix
import ood_scores as oods

BUNDLE = 'bundles/C4_seed42'
K, GATE1_PCT, GATE2_PCT = 10, 95, 80
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

cfg = json.load(open(f'{BUNDLE}/config.json'))
T, TAU = cfg['temperature'], cfg['decision_threshold']
FAR, NEAR, FLOOR = (cfg['thresholds'][k] for k in ('far_ood_knn', 'near_ood_knn', 'confidence_floor'))
REF = np.load(f'{BUNDLE}/gate_stats.npz')['ref_features'].astype(np.float64)
model = timm.create_model('efficientnet_b0', num_classes=2)
model.load_state_dict(torch.load(f'{BUNDLE}/weights.pt', map_location='cpu')); model.to(DEVICE).eval()
W2 = model.get_classifier().weight.detach().double().cpu().numpy()
B2 = model.get_classifier().bias.detach().double().cpu().numpy()
tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

@torch.no_grad()
def features(paths, bs=48):
    out = []
    for i in range(0, len(paths), bs):
        x = torch.stack([tf(Image.open(p).convert('RGB')) for p in paths[i:i + bs]]).to(DEVICE)
        out.append(model.forward_head(model.forward_features(x), pre_logits=True).float().cpu().numpy())
    return np.concatenate(out).astype(np.float64)

def probs(F, temperature=None):
    return oods.softmax((F @ W2.T + B2) / (T if temperature is None else temperature))[:, 1]

def novelty(F):
    return oods.knn_mean(F, REF, K)                                   # mean cosine distance to the 10 nearest reference lesions

def cascade(nov, conf, p, far=FAR, near=NEAR, floor=FLOOR, tau=TAU):
    """Gate 1 (far), Gate 2 (near), Gate 4 (confidence floor); Gate 3 is disabled. Returns
    the refusing gate ('' if answered), the answered mask and the call."""
    gate = np.full(len(nov), '', dtype=object)
    gate[nov > far] = 'far_ood'
    gate[(nov <= far) & (nov > near)] = 'near_ood'
    gate[(nov <= near) & (conf < floor)] = 'low_confidence'
    return gate, gate == '', np.where(p >= tau, 'melanoma', 'not melanoma')

def ece(p, y, bins=10):
    pred, conf = (p >= 0.5).astype(int), np.maximum(p, 1 - p)
    edges = np.linspace(0.5, 1.0, bins + 1); e = 0.0
    for b in range(bins):
        m = (conf >= edges[b]) & (conf < edges[b + 1] if b < bins - 1 else conf <= edges[b + 1])
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)

# ---------------------------------------------------------------- the test split
split = pd.read_csv('bundles/split.csv')                              # written by 04_train_classifier.py
train, cal, test = (split[split.part == p_] for p_ in ('train', 'calibration', 'test'))
F_TRAIN, F_CAL, F_TEST = (features(f.path.tolist()) for f in (train, cal, test))
P_CAL, NOV_CAL = probs(F_CAL), novelty(F_CAL); CONF_CAL = np.maximum(P_CAL, 1 - P_CAL)
Y = test.mel.values
P, NOV = probs(F_TEST), novelty(F_TEST); CONF = np.maximum(P, 1 - P)
GATE, ANS, CALL = cascade(NOV, CONF, P)

tn, fp, fn, tp = confusion_matrix(Y[ANS], (P[ANS] >= TAU).astype(int), labels=[0, 1]).ravel()
summary = dict(auroc=roc_auc_score(Y, P), ece=ece(P, Y), ece_before_temperature=ece(probs(F_TEST, 1.0), Y),
               coverage=ANS.mean(), sensitivity_answered=tp / (tp + fn), specificity_answered=tn / (tn + fp),
               sensitivity_all=((P >= TAU)[Y == 1]).mean(), melanomas_answered=int((ANS & (Y == 1)).sum()),
               refused_gate1=int((GATE == 'far_ood').sum()), refused_gate2=int((GATE == 'near_ood').sum()),
               refused_gate4=int((GATE == 'low_confidence').sum()))
print({k: round(float(v), 4) for k, v in summary.items()})

# contingency: answered/refused x melanoma x correct
rows = []
for name, yv in (('melanoma', 1), ('non-melanoma', 0)):
    m = Y == yv
    rows += [(name, 'answered, correct', int((ANS & m & ((CALL == 'melanoma') == bool(yv))).sum())),
             (name, 'answered, wrong', int((ANS & m & ((CALL == 'melanoma') != bool(yv))).sum()))]
    rows += [(name, f'refused, {g}', int((m & (GATE == g)).sum())) for g in ('far_ood', 'near_ood', 'low_confidence')]
print(pd.DataFrame(rows, columns=['truth', 'outcome', 'n']).pivot(index='truth', columns='outcome', values='n'))

# risk-coverage, ordered by confidence and by novelty
def risk_coverage(order):
    out = []
    for pct in (100, 95, 90, 85, 80, 70, 60, 50):
        keep = order[:int(len(order) * pct / 100)]
        pred = (P[keep] >= TAU).astype(int)
        out.append(dict(coverage=pct, risk=(pred != Y[keep]).mean(), recall=(pred[Y[keep] == 1] == 1).mean()))
    return pd.DataFrame(out)
print(risk_coverage(np.argsort(-CONF)).round(3)); print(risk_coverage(np.argsort(NOV)).round(3))

# what each gate contributes when applied alone, and cumulatively in order
rules = {'Gate 1': NOV > FAR, 'Gate 2': NOV > NEAR, 'Gate 4': CONF < FLOOR}
for g, fires in rules.items():
    others = np.any([v for k, v in rules.items() if k != g], axis=0)
    print(f'{g}: refuses {fires.sum()}, uniquely {(fires & ~others).sum()}, melanomas {(fires & (Y == 1)).sum()}')

# decision cut sweep (Figure 4.8)
cuts = np.linspace(0.01, 0.99, 99)
sens = [((P >= c)[Y == 1]).mean() for c in cuts]; spec = [((P < c)[Y == 0]).mean() for c in cuts]

# ---------------------------------------------------------------- outside populations
POPS = {   # name -> image paths; each is refused at the deployed thresholds and scored against the test split
    'MILK10k test, clinical close-up': glob.glob('MILK10k/clinical_closeups_of_test/*.jpg'),
    'Derm7pt, dermoscopic': glob.glob('Derm7pt/images/*/derm*.jpg'),
    'Derm7pt, clinical': glob.glob('Derm7pt/images/*/clinic*.jpg'),
    'DFUC2021, foot ulcers': sorted(glob.glob('DFUC2021/train/**/*.jpg', recursive=True)),
    'Oxford-IIIT pets': glob.glob('pets/*.jpg')[:400],
}
FEATS = {name: features(paths) for name, paths in POPS.items()}
for name, F in FEATS.items():
    nov = novelty(F)
    print(f'{name:34s} n {len(F):5d}  median novelty {np.median(nov):.3f}  refused {100 * (nov > NEAR).mean():5.1f}%  '
          f'AUROC vs test {oods.auroc(NOV, nov):.3f}')

# ---------------------------------------------------------------- the ulcer trial (Section 4.8)
F_DFU = FEATS['DFUC2021, foot ulcers']
p_d, nov_d = probs(F_DFU), novelty(F_DFU); conf_d = np.maximum(p_d, 1 - p_d)
g_d, a_d, c_d = cascade(nov_d, conf_d, p_d)
called = a_d & (c_d == 'melanoma')
print(f'ulcers: {len(F_DFU)}; refused Gate 1 {(g_d == "far_ood").sum()}, Gate 2 {(g_d == "near_ood").sum()}, '
      f'floor {(g_d == "low_confidence").sum()}; answered {a_d.sum()}; called melanoma {called.sum()} '
      f'({100 * called.mean():.1f}% of all, {100 * called.sum() / a_d.sum():.1f}% of answered); max p {p_d[a_d].max():.3f}')

def distinct(idx, F, thr=0.98):                                     # DFUC2021 has near-duplicate photographs
    U, kept = oods.unit(F[idx]), []
    for i, u in zip(idx, U):
        if all(float(u @ oods.unit(F[j:j + 1])[0]) < thr for j in kept):
            kept.append(i)
    return kept
print('called melanoma, distinct wounds:', len(distinct(np.where(called)[0], F_DFU)))

# everything later scripts need, cached once (the notebooks cached these on Drive)
np.savez_compressed('bundles/arrays.npz', F_TRAIN=F_TRAIN, F_CAL=F_CAL, F_TEST=F_TEST, Y=Y, Y_CAL=cal.mel.values,
                    P=P, P_RAW=probs(F_TEST, 1.0), NOV=NOV, CONF=CONF, P_CAL=P_CAL, NOV_CAL=NOV_CAL, CONF_CAL=CONF_CAL,
                    P_DFU=p_d, NOV_DFU=nov_d, GATE_DFU=g_d.astype(str), REF=REF, W2=W2, B2=B2,
                    **{'POP_' + name: F for name, F in FEATS.items()})
