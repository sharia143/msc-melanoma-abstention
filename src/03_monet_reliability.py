"""
How reliably MONET reads the seven-point checklist, scored against the dermatologist
labels of Derm7pt; reliability weights; conflict scores from expert and from MONET
concepts. Dissertation Sections 3.5 and 4.4 (Table 4.7, Figures 4.6-4.7).

Sharia Alam, MSc Data Science, Manchester Metropolitan University, 2026.
MONET is used exactly as released (suinleelab/monet); only the prompts are ours.
"""
import numpy as np, pandas as pd, torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

D7 = 'Derm7pt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
meta = pd.read_csv(f'{D7}/meta/meta.csv')
y = meta.diagnosis.str.startswith('melanoma').astype(int).values

# expert labels as binary criteria (the seven-point checklist)
E = np.stack([
    meta.pigment_network.eq('atypical'), meta.blue_whitish_veil.eq('present'),
    meta.vascular_structures.isin(['dotted', 'linear irregular']), meta.streaks.eq('irregular'),
    meta.pigmentation.isin(['diffuse irregular', 'localized irregular']),
    meta.dots_and_globules.eq('irregular'), meta.regression_structures.ne('absent')], 1).astype(float)
NAMES = ['atypical_pigment_network', 'blue_whitish_veil', 'atypical_vascular', 'irregular_streaks',
         'irregular_pigmentation', 'irregular_dots_globules', 'regression_structures']
POINTS = np.array([2, 2, 2, 1, 1, 1, 1])
assert ((E @ POINTS).astype(int) == meta.seven_point_score.values).all()   # the definitions reproduce the recorded score

PROMPTS = {   # (positive prompts, negative prompts); three of each per concept
    'atypical_pigment_network': (['dermoscopy of a melanocytic lesion with an atypical pigment network',
                                  'atypical pigment network with thickened lines and irregular holes',
                                  'irregular, non-uniform pigment network'],
                                 ['dermoscopy of a lesion with a typical regular pigment network',
                                  'uniform symmetric pigment network', 'no pigment network is present']),
    'blue_whitish_veil': (['dermoscopy showing a blue-whitish veil',
                           'confluent blue-white structureless area over the lesion', 'blue-white veil'],
                          ['dermoscopy with no blue-whitish veil', 'no blue-white structureless area',
                           'absence of a blue-white veil']),
    'atypical_vascular': (['dermoscopy showing an atypical vascular pattern',
                           'irregular linear vessels within the lesion', 'dotted vessels irregularly distributed'],
                          ['dermoscopy with no visible vascular structures', 'regular comma-shaped vessels',
                           'absence of atypical vessels']),
    'irregular_streaks': (['dermoscopy showing irregular streaks',
                           'irregular radial streaming and pseudopods at the lesion border',
                           'asymmetrically distributed streaks'],
                          ['dermoscopy with no streaks', 'regular streaks distributed symmetrically around the lesion',
                           'absence of radial streaming']),
    'irregular_pigmentation': (['dermoscopy showing irregular pigmentation',
                                'irregularly distributed pigmentation with abrupt cut-off',
                                'asymmetric blotchy pigmentation'],
                               ['dermoscopy with regular uniform pigmentation', 'evenly distributed pigmentation',
                                'no pigmentation']),
    'irregular_dots_globules': (['dermoscopy showing irregular dots and globules',
                                 'dots and globules of varying size irregularly distributed',
                                 'asymmetric dots and globules'],
                                ['dermoscopy with regular dots and globules',
                                 'uniformly sized and evenly spaced globules', 'no dots or globules']),
    'regression_structures': (['dermoscopy showing regression structures',
                               'white scar-like depigmentation within the lesion', 'blue-grey peppering or granularity'],
                              ['dermoscopy with no regression structures', 'no scar-like depigmentation',
                               'absence of peppering']),
}

processor = AutoProcessor.from_pretrained('suinleelab/monet')
monet = AutoModelForZeroShotImageClassification.from_pretrained('suinleelab/monet').to(DEVICE).eval()

@torch.no_grad()
def encode_images(paths, bs=32):
    out = []
    for i in range(0, len(paths), bs):
        px = processor(images=[Image.open(p).convert('RGB') for p in paths[i:i + bs]], return_tensors='pt')['pixel_values']
        f = monet.get_image_features(pixel_values=px.to(DEVICE))
        out.append(torch.nn.functional.normalize(f, dim=-1).cpu())
    return torch.cat(out)

@torch.no_grad()
def encode_text(texts):
    tk = processor(text=texts, return_tensors='pt', padding=True)
    return torch.nn.functional.normalize(monet.get_text_features(**{k: v.to(DEVICE) for k, v in tk.items()}), dim=-1).cpu()

IMF = encode_images([f'{D7}/images/{p}' for p in meta.derm])
V = np.stack([((IMF @ encode_text(pos).T).mean(1) - (IMF @ encode_text(neg).T).mean(1)).numpy()
              for pos, neg in (PROMPTS[n] for n in NAMES)], 1)     # one score per concept per image

auc = np.array([roc_auc_score(E[:, j], V[:, j]) for j in range(len(NAMES))])
W = np.clip(2 * auc - 1, 0, 1)                                      # reliability weight per concept
print(pd.DataFrame(dict(expert_prevalence=E.mean(0), monet_auroc=auc, weight=W), index=NAMES).round(3))
print(f'mean AUROC {auc.mean():.3f}')

# conflict scores from the two concept sources, and how far they agree
def zscore(M):
    return (M - M.mean(0)) / np.maximum(M.std(0), 1e-9)

def kappa(Z, weights=None, tau=1.0):
    w = np.ones(Z.shape[1]) if weights is None else weights
    D = ((Z[:, None, :] - Z[None, :, :]) ** 2 * w).sum(-1)
    K = np.exp(-D / (2 * tau ** 2)); np.fill_diagonal(K, 0.0)
    p = (K @ y) / np.maximum(K.sum(1), 1e-12)
    return np.minimum(p, 1 - p)

k_expert, k_monet = kappa(zscore(E)), kappa(zscore(V), W)
budget = 0.2                                                        # each source refuses its top 20%
ref_e = k_expert >= np.quantile(k_expert, 1 - budget)
ref_m = k_monet >= np.quantile(k_monet, 1 - budget)
from sklearn.metrics import cohen_kappa_score
print(f'agreement between the two refusal sets: Cohen kappa {cohen_kappa_score(ref_e, ref_m):.3f}')
print(f'melanomas answered confidently by MONET but ambiguous to experts: '
      f'{100 * (ref_e & ~ref_m & (y == 1)).sum() / (y == 1).sum():.1f}%')

# one row per case: expert criteria, MONET scores and the label, for the five-concept rerun (07_extensions.py)
pd.concat([pd.DataFrame(E, columns=[n + '_expert' for n in NAMES]), pd.DataFrame(V, columns=[n + '_monet' for n in NAMES]),
           pd.Series(y, name='is_mel')], axis=1).to_csv(f'{D7}/derm7pt_expert_and_monet_concepts.csv', index=False)
