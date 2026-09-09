"""
EfficientNet-B0 retrained with an eleven-class head, once on all MILK10k diagnoses and
once without each held-out type, to test whether a backbone that knows the lesion types
recognises an unfamiliar one. Dissertation Sections 3.7 and 4.9 (Figure 4.24, hypothesis H4).

Sharia Alam, MSc Data Science, Manchester Metropolitan University, 2026.
Same recipe as 04_train_classifier.py; early stopping on calibration macro recall.
"""
import os, json, random
import numpy as np, torch, timm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score
import ood_scores as oods
import importlib
base = importlib.import_module('04_train_classifier')            # split, transforms, LesionData, features, DEVICE, constants

SEED, K, OUT = 42, 10, 'bundles'
train, cal, test = base.train, base.cal, base.test
DIAG = sorted(train.dx.unique())

class MultiData(Dataset):
    def __init__(self, frame, tf, cidx):
        self.f, self.tf, self.c = frame.reset_index(drop=True), tf, cidx
    def __len__(self):
        return len(self.f)
    def __getitem__(self, i):
        r = self.f.iloc[i]
        return self.tf(base.Image.open(r.path).convert('RGB')), torch.tensor(self.c[r.dx])

def macro_recall(pred, true, n_cls):
    pred, true = np.asarray(pred), np.asarray(true)
    return float(np.mean([(pred[true == c] == c).mean() for c in range(n_cls) if (true == c).any()]))

def train_multiclass(holdout=None, seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    classes = [c for c in DIAG if c != holdout]; cidx = {c: i for i, c in enumerate(classes)}
    tr, ca = train[train.dx != holdout], cal[cal.dx != holdout]
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=len(classes)).to(base.DEVICE)
    cnt = tr.dx.map(cidx).value_counts().reindex(range(len(classes)), fill_value=1).values.astype(float)
    sampler = WeightedRandomSampler(torch.DoubleTensor((1.0 / cnt)[tr.dx.map(cidx).values]), len(tr), replacement=True)
    dl_tr = DataLoader(MultiData(tr, base.train_tf, cidx), batch_size=base.BATCH, sampler=sampler, num_workers=2)
    dl_ca = DataLoader(MultiData(ca, base.eval_tf, cidx), batch_size=64, num_workers=2)
    crit = torch.nn.CrossEntropyLoss(); opt = torch.optim.AdamW(model.parameters(), lr=base.LR, weight_decay=1e-4)
    best, best_state, bad, hist = -np.inf, None, 0, []
    for ep in range(base.EPOCHS):
        model.train()
        for x, y in dl_tr:
            x, y = x.to(base.DEVICE), y.to(base.DEVICE)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
        model.eval(); vp, vt = [], []
        with torch.no_grad():
            for x, y in dl_ca:
                vp += model(x.to(base.DEVICE)).argmax(1).cpu().tolist(); vt += y.tolist()
        mr = macro_recall(vp, vt, len(classes)); hist.append(mr)
        if mr > best + 1e-4:
            best, bad, best_state = mr, 0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= base.PATIENCE:
                break
    model.load_state_dict(best_state); model.eval()

    F_te = base.features(model, test)
    W, b = model.get_classifier().weight.detach().double().cpu().numpy(), model.get_classifier().bias.detach().double().cpu().numpy()
    lg = F_te @ W.T + b
    r = dict(held_out=holdout or 'none', seed=seed, n_classes=len(classes), best_epoch=int(np.argmax(hist)) + 1)
    if holdout:
        unseen = (test.dx == holdout).values
        F_tr = base.features(model, tr)
        rng = np.random.default_rng(seed); per = int(min(60, tr.dx.value_counts().min()))
        ref_i = np.concatenate([rng.choice(np.where(tr.dx.values == c)[0], min(per, (tr.dx.values == c).sum()), replace=False) for c in classes])
        R = oods.unit(F_tr[ref_i])
        for name, s in (('energy', oods.energy(lg)), ('msp', oods.msp(lg)), ('knn', oods.knn_mean(F_te, R, K))):
            r[f'auroc_{name}'] = oods.auroc(s[~unseen], s[unseen])
    else:
        pred, true = lg.argmax(1), test.dx.map(cidx).values
        r['test_macro_recall'] = macro_recall(pred, true, len(classes))
        r['melanoma_auroc_from_11class'] = float(roc_auc_score((test.dx == 'MEL').astype(int), oods.softmax(lg)[:, cidx['MEL']]))
    bundle = f'{OUT}/multi_{holdout or "all"}_seed{seed}'; os.makedirs(bundle, exist_ok=True)
    torch.save(model.state_dict(), f'{bundle}/weights.pt')
    json.dump(r, open(f'{bundle}/eval.json', 'w'), indent=2)
    return r

if __name__ == '__main__':
    for h in (None, 'BKL', 'SCCKA', 'AKIEC', 'NV'):
        print(train_multiclass(h))
