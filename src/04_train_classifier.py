"""
The abstaining melanoma classifier on MILK10k: the split, the four configurations
C1-C4, training with early stopping, and every deployment setting derived from the
calibration split (temperature, confidence floor, decision cut, novelty thresholds).
Dissertation Sections 3.1, 3.6 and 4.5 (Tables 3.1 and 4.9).

Sharia Alam, MSc Data Science, Manchester Metropolitan University, 2026.
"""
import os, json, random, time
import numpy as np, pandas as pd, torch, timm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import ood_scores as oods

MILK, IMGS, OUT = 'MILK10k', 'MILK10k/images', 'bundles'
SEED = 42
IMG, BATCH, LR, EPOCHS, PATIENCE = 224, 32, 3e-4, 10, 3
GATE1_PCT, GATE2_PCT, FLOOR_PCT, TARGET_SENS, K = 95, 80, 5, 0.85, 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CONFIGS = {   # the ablation of Table 4.9; C4 is the deployed system
    'C1': dict(sampler=False, weighted=False, temperature=False),
    'C2': dict(sampler=False, weighted=True,  temperature=False),
    'C3': dict(sampler=True,  weighted=False, temperature=False),
    'C4': dict(sampler=True,  weighted=False, temperature=True),
}

# ---------------------------------------------------------------- split: 70 / 15 / 15, stratified on melanoma
meta = pd.read_csv(f'{MILK}/MILK10k_Training_Metadata.csv')
gt = pd.read_csv(f'{MILK}/MILK10k_Training_GroundTruth.csv')
DIAG = [c for c in gt.columns if c != 'lesion_id']
d = meta[meta.image_type == 'dermoscopic'].merge(gt, on='lesion_id')
d['path'] = [os.path.join(IMGS, l, i + '.jpg') for l, i in zip(d.lesion_id, d.isic_id)]
d['mel'], d['dx'] = d.MEL.astype(int), d[DIAG].idxmax(axis=1)
train, rest = train_test_split(d, test_size=0.30, stratify=d.mel, random_state=SEED)
cal, test = train_test_split(rest, test_size=0.50, stratify=rest.mel, random_state=SEED)
assert len(d) == 5240 and int(test.mel.sum()) == 67
os.makedirs(OUT, exist_ok=True)
pd.concat([train.assign(part='train'), cal.assign(part='calibration'), test.assign(part='test')])[
    ['lesion_id', 'isic_id', 'path', 'mel', 'dx', 'skin_tone_class', 'sex', 'part']].to_csv(f'{OUT}/split.csv', index=False)

# ---------------------------------------------------------------- data and model
norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tf = transforms.Compose([transforms.Resize((IMG, IMG)), transforms.RandomHorizontalFlip(),
                               transforms.RandomVerticalFlip(), transforms.ColorJitter(0.1, 0.1, 0.1),
                               transforms.ToTensor(), norm])
eval_tf = transforms.Compose([transforms.Resize((IMG, IMG)), transforms.ToTensor(), norm])

class LesionData(Dataset):
    def __init__(self, frame, tf):
        self.f, self.tf = frame.reset_index(drop=True), tf
    def __len__(self):
        return len(self.f)
    def __getitem__(self, i):
        r = self.f.iloc[i]
        return self.tf(Image.open(r.path).convert('RGB')), torch.tensor(int(r.mel))

def balanced_accuracy(pred, true):
    pred, true = np.asarray(pred), np.asarray(true)
    return ((pred[true == 1] == 1).mean() + (pred[true == 0] == 0).mean()) / 2

@torch.no_grad()
def logits(model, frame):
    out, ys = [], []
    for x, y in DataLoader(LesionData(frame, eval_tf), batch_size=64, num_workers=2):
        out.append(model(x.to(DEVICE)).float().cpu().numpy()); ys += y.tolist()
    return np.concatenate(out).astype(np.float64), np.array(ys)

@torch.no_grad()
def features(model, frame):
    out = []
    for x, _ in DataLoader(LesionData(frame, eval_tf), batch_size=64, num_workers=2):
        out.append(model.forward_head(model.forward_features(x.to(DEVICE)), pre_logits=True).float().cpu().numpy())
    return np.concatenate(out).astype(np.float64)                   # 1280-d penultimate features

def ece(p, y, bins=10):
    pred, conf = (p >= 0.5).astype(int), np.maximum(p, 1 - p)
    edges = np.linspace(0.5, 1.0, bins + 1); e = 0.0
    for b in range(bins):
        m = (conf >= edges[b]) & (conf < edges[b + 1] if b < bins - 1 else conf <= edges[b + 1])
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)

def fit_temperature(lg, y):
    def nll(T):
        z = lg / T; z = z - z.max(1, keepdims=True)
        return -(z - np.log(np.exp(z).sum(1, keepdims=True)))[np.arange(len(y)), y].mean()
    grid = np.exp(np.linspace(np.log(0.25), np.log(20), 400))
    return float(grid[int(np.argmin([nll(t) for t in grid]))])

# ---------------------------------------------------------------- one configuration, one seed
def train_config(cfg, seed):
    rc = CONFIGS[cfg]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2).to(DEVICE)
    cnt = np.bincount(train.mel.values, minlength=2)
    if rc['sampler']:
        sampler = WeightedRandomSampler(torch.DoubleTensor((1.0 / cnt)[train.mel.values]), len(train), replacement=True)
        dl_tr = DataLoader(LesionData(train, train_tf), batch_size=BATCH, sampler=sampler, num_workers=2)
    else:
        dl_tr = DataLoader(LesionData(train, train_tf), batch_size=BATCH, shuffle=True, num_workers=2)
    dl_ca = DataLoader(LesionData(cal, eval_tf), batch_size=64, num_workers=2)
    weight = torch.tensor(cnt.sum() / (2.0 * cnt), dtype=torch.float32).to(DEVICE) if rc['weighted'] else None
    crit = torch.nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=DEVICE == 'cuda')

    best, best_state, bad, history = -np.inf, None, 0, []
    for ep in range(EPOCHS):
        model.train()
        for x, y in dl_tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.amp.autocast('cuda', enabled=DEVICE == 'cuda'):
                loss = crit(model(x), y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        model.eval(); vp, vt = [], []
        with torch.no_grad():
            for x, y in dl_ca:
                vp += model(x.to(DEVICE)).argmax(1).cpu().tolist(); vt += y.tolist()
        bal = balanced_accuracy(vp, vt); history.append(bal)
        if bal > best + 1e-4:                                        # early stopping on calibration balanced accuracy
            best, bad = bal, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state); model.eval()

    # every deployment setting comes from the calibration split, by the same rule for all configurations
    lg_cal, y_cal = logits(model, cal)
    T = fit_temperature(lg_cal, y_cal) if rc['temperature'] else 1.0
    p_cal = oods.softmax(lg_cal / T)[:, 1]
    floor = float(np.percentile(np.maximum(p_cal, 1 - p_cal), FLOOR_PCT))
    cuts = np.round(np.linspace(0.001, 0.999, 999), 4)
    reach = [c for c in cuts if ((p_cal >= c)[y_cal == 1]).mean() >= TARGET_SENS]
    tau = float(max(reach)) if reach else 0.5                        # loosest cut reaching the target sensitivity
    F_tr, F_cal = features(model, train), features(model, cal)
    mel_idx, ben_idx = np.where(train.mel.values == 1)[0], np.where(train.mel.values == 0)[0]
    rng = np.random.default_rng(seed)
    REF = oods.unit(F_tr[np.r_[rng.choice(ben_idx, len(mel_idx), replace=False), mel_idx]])   # balanced reference, 630 lesions
    nov_cal = oods.knn_mean(F_cal, REF, K)
    thr_far, thr_near = float(np.percentile(nov_cal, GATE1_PCT)), float(np.percentile(nov_cal, GATE2_PCT))

    # the test split through this model, at this model's own settings
    lg_te, y_te = logits(model, test)
    p_te = oods.softmax(lg_te / T)[:, 1]; conf_te = np.maximum(p_te, 1 - p_te)
    nov_te = oods.knn_mean(features(model, test), REF, K)
    refused = (nov_te > thr_near) | (conf_te < floor)
    ans = ~refused; call = p_te >= tau
    tp = int((ans & call & (y_te == 1)).sum()); fn = int((ans & ~call & (y_te == 1)).sum())
    tn = int((ans & ~call & (y_te == 0)).sum()); fp = int((ans & call & (y_te == 0)).sum())
    evaluation = dict(config=cfg, seed=seed, T=T, tau=tau, floor=floor, far=thr_far, near=thr_near,
                      best_epoch=int(np.argmax(history)) + 1, auroc=float(roc_auc_score(y_te, p_te)),
                      ece=ece(p_te, y_te), coverage=float(ans.mean()), melanomas_answered=int((ans & (y_te == 1)).sum()),
                      sensitivity_answered=tp / max(tp + fn, 1), specificity_answered=tn / max(tn + fp, 1),
                      sensitivity_all=float((p_te >= tau)[y_te == 1].mean()))

    bundle = f'{OUT}/{cfg}_seed{seed}'; os.makedirs(bundle, exist_ok=True)
    torch.save(model.state_dict(), f'{bundle}/weights.pt')
    np.savez_compressed(f'{bundle}/gate_stats.npz', ref_features=REF.astype(np.float32))
    np.savez_compressed(f'{bundle}/test_scores.npz', p=p_te, nov=nov_te, conf=conf_te, y=y_te, answered=ans, call=call)
    json.dump(dict(arch='efficientnet_b0', img_size=IMG, knn_k=K, temperature=T, decision_threshold=tau, imbalance=cfg,
                   best_epoch=evaluation['best_epoch'],
                   thresholds=dict(far_ood_knn=thr_far, near_ood_knn=thr_near, confidence_floor=floor)),
              open(f'{bundle}/config.json', 'w'), indent=2)
    json.dump(evaluation, open(f'{bundle}/eval.json', 'w'), indent=2)
    return evaluation

if __name__ == '__main__':
    for cfg in CONFIGS:
        for seed in (42, 7, 2026):
            t0 = time.time(); out = train_config(cfg, seed)
            print(f'{(time.time() - t0) / 60:.1f} min', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()})
# deployed settings (C4, seed 42): T = 5.068, cut 0.350, floor 0.5506, thresholds 0.7883 / 0.7154
