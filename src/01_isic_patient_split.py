"""
ISIC-DICM-17K: released split versus a patient-aware split, and the four baselines.
Dissertation Sections 3.1-3.2 and 4.1 (Tables 4.1-4.2, Figures 4.1-4.4).

Sharia Alam, MSc Data Science, Manchester Metropolitan University, 2026.
Run in Google Colab; IMG_DIR must point at the ISIC-DICM-17K images.
"""
import glob, os, random
import numpy as np, pandas as pd, torch, timm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

SEED, EPOCHS, IMG_SIZE = 42, 5, 224
IMG_DIR = '/content/drive/MyDrive/MSc_Thesis_ISIC/images'
META = 'https://raw.githubusercontent.com/mmu-dermatology-research/isic-dicm-17k/main'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---------------------------------------------------------------- data
train = pd.read_csv(f'{META}/train-set-metadata.csv')
valid = pd.read_csv(f'{META}/valid-set-metadata.csv')

files = [f for e in ('*.jpg', '*.jpeg', '*.png', '*.JPG') for f in glob.glob(os.path.join(IMG_DIR, '**', e), recursive=True)]
by_name = {os.path.basename(f): f for f in files}

def resolve(row):
    for cand in (row['image'], f"{row['isic_id']}.jpg"):
        if cand in by_name:
            return by_name[cand]
    return None

for d in (train, valid):
    d['path'] = d.apply(resolve, axis=1)
train, valid = train[train.path.notna()], valid[valid.path.notna()]

# metadata completeness (Figure 4.2): only age, sex and site are usable inputs
missing = pd.concat([train, valid]).isna().mean().sort_values(ascending=False)
print(missing.round(3))

# ---------------------------------------------------------------- the two splits
full = pd.concat([train, valid], ignore_index=True)
full['grp'] = full.patient_id.fillna(pd.Series([f'_solo{i}' for i in range(len(full))]))
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
tr_idx, va_idx = next(sgkf.split(full, full['class'], groups=full.grp))
paw_tr, paw_va = full.iloc[tr_idx], full.iloc[va_idx]

def shared_patients(a, b):
    shared = set(a.patient_id.dropna()) & set(b.patient_id.dropna())
    rows = b.patient_id.isin(shared).sum()
    return len(shared), rows, 100 * rows / len(b)

for name, (a, b) in {'released': (train, valid), 'patient-aware': (paw_tr, paw_va)}.items():
    n, rows, pct = shared_patients(a, b)
    print(f'{name:14s} train {len(a):6d} val {len(b):6d}  shared patients {n:4d} -> {rows} val rows ({pct:.1f}%)')

# ---------------------------------------------------------------- training
norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.RandomHorizontalFlip(),
                               transforms.RandomVerticalFlip(), transforms.ColorJitter(0.1, 0.1, 0.1),
                               transforms.ToTensor(), norm])
eval_tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), norm])

class DS(Dataset):
    def __init__(self, frame, tf):
        self.f, self.tf = frame.reset_index(drop=True), tf
    def __len__(self):
        return len(self.f)
    def __getitem__(self, i):
        r = self.f.iloc[i]
        return self.tf(Image.open(r.path).convert('RGB')), torch.tensor(int(r['class']))

def ece(p, y, pred, bins=10):
    # binned on confidence max(p, 1-p); the earlier version binned on p and was wrong
    conf = np.maximum(p, 1 - p); edges = np.linspace(0.5, 1.0, bins + 1); e = 0.0
    for b in range(bins):
        m = (conf >= edges[b]) & (conf < edges[b + 1] if b < bins - 1 else conf <= edges[b + 1])
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return e

def train_and_eval(df_tr, df_va, name, arch='efficientnet_b0'):
    torch.manual_seed(SEED)
    dl_tr = DataLoader(DS(df_tr, train_tf), batch_size=32, shuffle=True, num_workers=2)
    dl_va = DataLoader(DS(df_va, eval_tf), batch_size=64, num_workers=2)
    model = timm.create_model(arch, pretrained=True, num_classes=2).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    crit = torch.nn.CrossEntropyLoss()
    for _ in range(EPOCHS):
        model.train()
        for x, y in dl_tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
    model.eval(); P, Y = [], []
    with torch.no_grad():
        for x, y in dl_va:
            P += F.softmax(model(x.to(DEVICE)), 1)[:, 1].cpu().tolist(); Y += y.tolist()
    P, Y = np.array(P), np.array(Y); pred = (P >= 0.5).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(Y, pred, average='binary')
    return dict(split=name, accuracy=accuracy_score(Y, pred), precision=pr, recall=rc, f1=f1,
                auroc=roc_auc_score(Y, P), ece=ece(P, Y, pred), brier=float(np.mean((P - Y) ** 2)))

results = pd.DataFrame([train_and_eval(train, valid, 'released'),
                        train_and_eval(paw_tr, paw_va, 'patient-aware')]).set_index('split')
print(results.round(3))
print('inflation, released minus patient-aware:')
print((results.loc['released'] - results.loc['patient-aware']).round(3))

# ViT and zero-shot BiomedCLIP baselines (Table 4.2) used the same loop with
# arch='vit_base_patch16_224' and, for BiomedCLIP, open_clip zero-shot prompts.
