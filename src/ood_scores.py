"""
ood_scores.py -- OOD / abstention scores computed on frozen features
Sharia Alam, MSc Data Science, Manchester Metropolitan University

Alternative scores for the novelty gates, all computed from the same saved
1280-d features and the same classifier head as the deployed cascade
(06_alternative_scores.py). Nothing here retrains the network.

Convention: every score is an OOD score, higher = more novel, so the rule
"refuse if score > threshold" reads the same way for all of them.

    knn_mean        mean cosine distance to the k nearest reference features
                    (the deployed gate; variant of Sun et al. 2022)
    knn_kth         cosine distance to the k-th nearest reference feature
                    (Sun et al. 2022 as published)
    msp             1 - max softmax probability      (Hendrycks & Gimpel 2017)
    max_logit       - max logit                      (Hendrycks et al. 2022)
    energy          - logsumexp(logits)              (Liu et al. 2020, T = 1)
    Mahalanobis     min_k Mahalanobis distance to class k, tied covariance
                    (Lee et al. 2018; single layer, no input pre-processing)
    Mahalanobis.rmd relative Mahalanobis, min_k (MD_k - MD_0)   (Ren et al. 2021)
    ViM             alpha * ||residual|| - logsumexp(logits)    (Wang et al. 2022)
    LinearProbe     multi-class logistic-regression head on the frozen features,
                    so msp / max_logit / energy / ViM can be read from an
                    11-class head instead of the binary one
    crc_threshold   conformal risk control            (Angelopoulos et al. 2022)

numpy / scipy / scikit-learn only; `python ood_scores.py --selftest` runs the checks.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp as _lse


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def unit(F):
    """Rows scaled to unit length."""
    F = np.asarray(F, dtype=np.float64)
    return F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-12)


def softmax(logits):
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def auroc(id_scores, ood_scores):
    """AUROC for separating OOD (positive) from ID, with higher = more OOD."""
    from sklearn.metrics import roc_auc_score
    s = np.r_[np.asarray(id_scores, float), np.asarray(ood_scores, float)]
    y = np.r_[np.zeros(len(id_scores)), np.ones(len(ood_scores))]
    return float(roc_auc_score(y, s))


def percentile_of(values, reference):
    """Where each value falls in a reference sample, as 0-100."""
    ref = np.sort(np.asarray(reference, float))
    return 100.0 * np.searchsorted(ref, np.asarray(values, float)) / max(len(ref), 1)


# --------------------------------------------------------------------------
# feature-space scores
# --------------------------------------------------------------------------

def knn_mean(F, ref_unit, k=10):
    """The deployed score: 1 - mean of the k largest cosine similarities."""
    sims = unit(F) @ np.asarray(ref_unit, np.float64).T
    return 1.0 - np.sort(sims, axis=1)[:, -k:].mean(axis=1)


def knn_kth(F, ref_unit, k=10):
    """Sun et al. (2022): 1 - the k-th largest cosine similarity."""
    sims = unit(F) @ np.asarray(ref_unit, np.float64).T
    return 1.0 - np.sort(sims, axis=1)[:, -k]


class Mahalanobis:
    """Class-conditional Mahalanobis distance with a tied Ledoit-Wolf covariance.

    fit(F, y) stores the class means, one shrunk covariance of the centred
    features, and a background Gaussian on all features (needed for rmd).
    score() = min_k MD_k(z) (Lee et al. 2018); rmd() = min_k [MD_k(z) - MD_0(z)]
    (Ren et al. 2021).
    """

    def fit(self, F, y):
        from sklearn.covariance import LedoitWolf
        F = np.asarray(F, np.float64); y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.means_ = np.stack([F[y == c].mean(0) for c in self.classes_])
        centred = F - self.means_[np.searchsorted(self.classes_, y)]
        lw = LedoitWolf().fit(centred)
        P = lw.precision_
        self.precision_ = (P + P.T) / 2.0
        lw0 = LedoitWolf().fit(F)
        self.mean0_ = F.mean(0)
        P0 = lw0.precision_
        self.precision0_ = (P0 + P0.T) / 2.0
        return self

    def _md(self, F, mean, precision):
        d = np.asarray(F, np.float64) - mean
        return np.einsum("ij,jk,ik->i", d, precision, d)

    def per_class(self, F):
        return np.stack([self._md(F, m, self.precision_) for m in self.means_], 1)

    def score(self, F):
        return self.per_class(F).min(axis=1)

    def rmd(self, F):
        md0 = self._md(F, self.mean0_, self.precision0_)
        return (self.per_class(F) - md0[:, None]).min(axis=1)


# --------------------------------------------------------------------------
# logit-space scores
# --------------------------------------------------------------------------

def msp(logits):
    """1 - max softmax probability."""
    return 1.0 - softmax(logits).max(axis=1)


def max_logit(logits):
    return -np.asarray(logits, np.float64).max(axis=1)


def energy(logits, T=1.0):
    """Liu et al. (2020): E(x) = -T logsumexp(f(x)/T). Higher = more OOD."""
    z = np.asarray(logits, np.float64) / T
    return -T * _lse(z, axis=1)


class ViM:
    """Virtual-logit matching (Wang et al., CVPR 2022).

    fit(F, W, b) takes training features (N, D) and the head's weight (C, D)
    and bias (C,). score(z) = alpha * ||residual(z)|| - logsumexp(W z + b),
    higher = more OOD. n_principal (D') defaults to the rule in the authors'
    code: 1000 for D >= 2048, 512 for 768 <= D < 2048, else D // 2; for
    EfficientNet-B0 (D = 1280) that is 512.
    """

    def __init__(self, n_principal=None):
        self.n_principal = n_principal

    def fit(self, F, W, b):
        F = np.asarray(F, np.float64)
        W = np.asarray(W, np.float64)          # (C, D)
        b = np.asarray(b, np.float64)          # (C,)
        D = F.shape[1]
        if self.n_principal is None:
            self.n_principal = 1000 if D >= 2048 else 512 if D >= 768 else D // 2
        self.n_principal = int(min(self.n_principal, D - 1, F.shape[0] - 1))
        self.W_, self.b_ = W, b
        # offset u = -(W^T)^+ b puts the origin where the logits vanish
        self.u_ = -np.linalg.pinv(W) @ b                    # (D,)
        Fc = F - self.u_
        # principal subspace: leading eigenvectors of Fc^T Fc, largest first
        cov = Fc.T @ Fc
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        self.P_ = vecs[:, order[:self.n_principal]]         # (D, D')
        # alpha scales the residual norms so their training-set sum matches
        # the sum of the max logits
        resid = self._residual_norm(F)
        logits = F @ W.T + b
        self.alpha_ = float(logits.max(axis=1).sum() / max(resid.sum(), 1e-12))
        return self

    def _residual_norm(self, F):
        Fc = np.asarray(F, np.float64) - self.u_
        proj = Fc @ self.P_ @ self.P_.T
        return np.linalg.norm(Fc - proj, axis=1)

    def score(self, F, logits=None):
        F = np.asarray(F, np.float64)
        if logits is None:
            logits = F @ self.W_.T + self.b_
        vlogit = self.alpha_ * self._residual_norm(F)
        return vlogit - _lse(np.asarray(logits, np.float64), axis=1)


# --------------------------------------------------------------------------
# a multi-class head on the frozen features
# --------------------------------------------------------------------------

class LinearProbe:
    """Multinomial logistic regression on standardised frozen features.

    Gives a K-class head whose logits can feed msp / max_logit / energy / ViM
    without retraining the backbone.
    """

    def __init__(self, C=0.5, max_iter=3000, seed=42):
        self.C, self.max_iter, self.seed = C, max_iter, seed

    def fit(self, F, y):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        F = np.asarray(F, np.float64)
        self.scaler_ = StandardScaler().fit(F)
        self.clf_ = LogisticRegression(C=self.C, max_iter=self.max_iter,
                                       random_state=self.seed)
        self.clf_.fit(self.scaler_.transform(F), np.asarray(y))
        self.classes_ = self.clf_.classes_
        return self

    def transform(self, F):
        return self.scaler_.transform(np.asarray(F, np.float64))

    def logits(self, F):
        Z = self.transform(F)
        lg = Z @ self.clf_.coef_.T + self.clf_.intercept_
        if lg.shape[1] == 1:                      # binary sklearn convention
            lg = np.c_[-lg[:, 0] / 2, lg[:, 0] / 2]
        return lg

    def predict(self, F):
        return self.classes_[self.logits(F).argmax(axis=1)]

    def accuracy(self, F, y):
        return float((self.predict(F) == np.asarray(y)).mean())

    def vim(self, F_train, n_principal=None):
        """A ViM scorer in the probe's standardised feature space."""
        W = self.clf_.coef_
        b = self.clf_.intercept_
        if W.shape[0] == 1:
            W = np.r_[-W / 2, W / 2]; b = np.r_[-b / 2, b / 2]
        v = ViM(n_principal=n_principal).fit(self.transform(F_train), W, b)
        probe = self

        class _Wrapped:
            def score(self, F):
                return v.score(probe.transform(F))
        return _Wrapped()


def loco_probe(F_train, dx_train, F_test, dx_test, holdouts, min_test=20,
               knn_ref_unit=None, k=10, C=0.5, seed=42):
    """Leave-one-class-out with a linear probe on frozen features.

    For each held-out class c, fit a probe on the training lesions of the other
    classes and score the test lesions of c against the test lesions of the
    known classes. Returns one dict per class with the AUROC of msp / max_logit
    / energy (and of knn_mean when a reference set is given). Note the backbone
    has already seen class c during binary training, so this tests the head only.
    """
    rows = []
    dx_train = np.asarray(dx_train); dx_test = np.asarray(dx_test)
    for c in holdouts:
        unseen = dx_test == c
        if unseen.sum() < min_test:
            continue
        keep_tr = dx_train != c
        probe = LinearProbe(C=C, seed=seed).fit(F_train[keep_tr], dx_train[keep_tr])
        known = ~unseen
        lg_known, lg_unseen = probe.logits(F_test[known]), probe.logits(F_test[unseen])
        row = dict(held_out=c, n_unseen=int(unseen.sum()), n_known=int(known.sum()),
                   msp=auroc(msp(lg_known), msp(lg_unseen)),
                   max_logit=auroc(max_logit(lg_known), max_logit(lg_unseen)),
                   energy=auroc(energy(lg_known), energy(lg_unseen)))
        if knn_ref_unit is not None:
            row["knn_mean"] = auroc(knn_mean(F_test[known], knn_ref_unit, k),
                                    knn_mean(F_test[unseen], knn_ref_unit, k))
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# operating points
# --------------------------------------------------------------------------

def refusal_at_quantile(cal_scores, pop_scores, pct):
    """Threshold at the pct-th percentile of calibration scores; refusal rate."""
    thr = float(np.percentile(np.asarray(cal_scores, float), pct))
    return thr, float((np.asarray(pop_scores, float) > thr).mean())


def rank_fusion(score_a, score_b, cal_a, cal_b, w=0.5):
    """Average of calibration percentiles of two scores (a scale-free fusion)."""
    return w * percentile_of(score_a, cal_a) + (1 - w) * percentile_of(score_b, cal_b)


# --------------------------------------------------------------------------
# conformal risk control
# --------------------------------------------------------------------------

def crc_threshold(losses, lambdas, alpha, B=1.0):
    """Conformal risk control (Angelopoulos et al. 2022).

    losses : (n, L) array, column j = loss of calibration point i at lambdas[j];
             losses must be bounded by B and non-increasing in j.
    returns (j_hat, lambdas[j_hat]), the first lambda with
             n/(n+1) * mean_i L_i(lambda) + B/(n+1) <= alpha,
             or (None, None) if no lambda satisfies it. Guarantee:
             E[L(lambda_hat)] <= alpha on a new exchangeable point.
    """
    losses = np.asarray(losses, float)
    n = losses.shape[0]
    risk = losses.mean(axis=0)
    if np.any(np.diff(risk) > 1e-12):
        raise ValueError("losses must be non-increasing in lambda for CRC")
    bound = n / (n + 1) * risk + B / (n + 1)
    ok = np.where(bound <= alpha)[0]
    if len(ok) == 0:
        return None, None
    j = int(ok[0])
    return j, float(np.asarray(lambdas, float)[j])


def miss_losses_floor(p_mel, conf, floors, tau, keep=None):
    """Loss matrix for CRC on the confidence floor, melanoma cases only.

    L_i(floor) = 1 if the case is answered (conf >= floor) and missed (p < tau);
    raising the floor can only turn answers into refusals, so the loss is
    non-increasing in the floor. `keep` masks the cases the novelty gates let
    through (a refused case cannot be missed).
    """
    p_mel = np.asarray(p_mel, float); conf = np.asarray(conf, float)
    floors = np.asarray(floors, float)
    missed = (p_mel < tau)[:, None]
    answered = conf[:, None] >= floors[None, :]
    L = (missed & answered).astype(float)
    if keep is not None:
        L = L * np.asarray(keep, float)[:, None]
    return L


def miss_losses_cut(p_mel, cuts, keep=None):
    """Loss matrix for CRC on the decision cut, melanoma cases only.

    L_i(cut) = 1 if p_i < cut. Pass `cuts` ordered from high to low so the loss
    is non-increasing. `keep` masks the cases the cascade answers.
    """
    p_mel = np.asarray(p_mel, float); cuts = np.asarray(cuts, float)
    L = (p_mel[:, None] < cuts[None, :]).astype(float)
    if keep is not None:
        L = L * np.asarray(keep, float)[:, None]
    return L


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def selftest(verbose=True):
    rng = np.random.default_rng(0)
    D, K = 64, 4
    ok = True

    def check(label, cond):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {label}")

    # in-distribution: K class clusters; near-OOD: a fifth cluster between them;
    # far-OOD: isotropic noise far away
    means = rng.normal(size=(K, D)) * 3.0
    y_tr = rng.integers(0, K, 1500); F_tr = means[y_tr] + rng.normal(size=(1500, D))
    y_te = rng.integers(0, K, 600);  F_te = means[y_te] + rng.normal(size=(600, D))
    F_near = means.mean(0) + rng.normal(size=(300, D)) * 1.5
    F_far = rng.normal(size=(300, D)) * 6 + 20

    ref = unit(F_tr)
    for name, fn in [("knn_mean", knn_mean), ("knn_kth", knn_kth)]:
        a_far = auroc(fn(F_te, ref), fn(F_far, ref))
        check(f"{name}: far-OOD AUROC {a_far:.3f} > 0.95", a_far > 0.95)

    maha = Mahalanobis().fit(F_tr, y_tr)
    check(f"Mahalanobis far AUROC {auroc(maha.score(F_te), maha.score(F_far)):.3f} > 0.95",
          auroc(maha.score(F_te), maha.score(F_far)) > 0.95)
    a_rmd = auroc(maha.rmd(F_te), maha.rmd(F_near))
    a_md = auroc(maha.score(F_te), maha.score(F_near))
    check(f"RMD near-OOD AUROC {a_rmd:.3f} >= 0.80", a_rmd >= 0.80)
    if verbose:
        print(f"         (plain Mahalanobis on the same near-OOD set: {a_md:.3f})")

    probe = LinearProbe(C=1.0).fit(F_tr, y_tr)
    check(f"probe accuracy {probe.accuracy(F_te, y_te):.3f} > 0.95",
          probe.accuracy(F_te, y_te) > 0.95)
    lg_te, lg_near = probe.logits(F_te), probe.logits(F_near)
    for name, fn in [("msp", msp), ("max_logit", max_logit), ("energy", energy)]:
        a = auroc(fn(lg_te), fn(lg_near))
        check(f"{name} on probe logits, near-OOD AUROC {a:.3f} > 0.85", a > 0.85)

    W = probe.clf_.coef_; b = probe.clf_.intercept_
    vim = ViM(n_principal=16).fit(probe.transform(F_tr), W, b)
    a_vim = auroc(vim.score(probe.transform(F_te)), vim.score(probe.transform(F_far)))
    check(f"ViM far-OOD AUROC {a_vim:.3f} > 0.95", a_vim > 0.95)
    check("ViM alpha is positive and finite",
          np.isfinite(vim.alpha_) and vim.alpha_ > 0)

    # energy equals -logsumexp
    lg = rng.normal(size=(5, 3))
    check("energy(T=1) == -logsumexp", np.allclose(energy(lg), -_lse(lg, axis=1)))
    # msp in [0, 1)
    check("msp within [0,1)", np.all((msp(lg) >= 0) & (msp(lg) < 1)))

    # LOCO with the probe: the held-out class should be detectable
    rows = loco_probe(F_tr, y_tr.astype(str), F_te, y_te.astype(str),
                      holdouts=["0", "1"], knn_ref_unit=ref)
    check("loco_probe returns one row per held-out class", len(rows) == 2)
    check(f"loco_probe energy AUROC {rows[0]['energy']:.3f} > 0.7", rows[0]["energy"] > 0.7)

    # CRC: the guarantee holds in expectation on fresh draws
    p_cal = rng.beta(4, 2, 200); conf_cal = np.maximum(p_cal, 1 - p_cal)
    floors = np.linspace(0.5, 1.0, 101)
    L = miss_losses_floor(p_cal, conf_cal, floors, tau=0.35)
    j, lam = crc_threshold(L, floors, alpha=0.10)
    check("CRC finds a floor", j is not None)
    realised = []
    for _ in range(200):
        p_new = rng.beta(4, 2, 200); c_new = np.maximum(p_new, 1 - p_new)
        realised.append(miss_losses_floor(p_new, c_new, np.array([lam]), 0.35).mean())
    check(f"CRC realised risk {np.mean(realised):.3f} <= 0.10", np.mean(realised) <= 0.10)
    cuts = np.linspace(0.9, 0.05, 86)           # high -> low, so loss is non-increasing
    Lc = miss_losses_cut(p_cal, cuts)
    j2, cut_hat = crc_threshold(Lc, cuts, alpha=0.15)
    check(f"CRC on the cut returns a cut ({cut_hat})", j2 is not None)
    try:
        crc_threshold(Lc[:, ::-1], cuts[::-1], alpha=0.15)
        check("CRC rejects increasing losses", False)
    except ValueError:
        check("CRC rejects increasing losses", True)

    print("\nSELF-TEST " + ("PASSED" if ok else "FAILED"))
    return ok


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(0 if selftest() else 1)
    print(__doc__)
