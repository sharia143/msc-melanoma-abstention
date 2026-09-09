# Concept inconsistency and abstention in melanoma classification

Code behind the MSc Data Science dissertation of Sharia Alam (Manchester Metropolitan
University, 2026), supervised by Prof. Moi Hoon Yap. The project asks when a melanoma
classifier should decline to answer: it reproduces the concept-inconsistency analysis of
Nápoles et al. (2026) on Derm7pt, extends it to MILK10k, measures how reliably MONET reads
the seven-point checklist, and builds an EfficientNet-B0 classifier with a cascade of
abstention gates that is then tested on clinical photographs, a second dermoscopic
collection, diabetic foot ulcers and photographs that are not skin.

## What is here

The scripts are the working code of the project, condensed from the Colab notebooks in
which the experiments were run, in the order the dissertation reports them.

| script | dissertation |
|---|---|
| `src/01_isic_patient_split.py` | patient leakage in the released ISIC-DICM-17K split; the patient-aware split; baselines (§4.1) |
| `src/02_concept_inconsistency.py` | rough-set profile analysis on Derm7pt and MILK10k; concept-conflict abstention and its reduction property (§3.3–3.4, §4.2–4.3) |
| `src/03_monet_reliability.py` | MONET concept scores against dermatologist labels; reliability weights; conflict from expert and from MONET concepts (§3.5, §4.4) |
| `src/04_train_classifier.py` | the MILK10k split; configurations C1–C4; training; temperature, floor, cut and novelty thresholds from the calibration split (§3.6, §4.5) |
| `src/05_abstaining_cascade.py` | the cascade on the test split; calibration; risk–coverage; gate contributions; refusal by population; the ulcer trial (§4.5–4.8) |
| `src/06_alternative_scores.py` | twelve alternative scores on the same features; leave-one-class-out with a probe; conformal risk control (§4.9–4.10) |
| `src/07_extensions.py` | seeds, bootstrap intervals, skin-tone and sex audit, wound-aware stage, kNN sensitivity, five-concept rerun, before/after (§4.3, §4.5, §4.7, §4.11–4.13) |
| `src/08_eleven_class_backbone.py` | the backbone retrained with eleven classes under leave-one-class-out (§4.9) |
| `src/ood_scores.py` | the abstention scores (kNN, Mahalanobis, MSP, max logit, energy, ViM, linear probe) and conformal risk control, as functions |
| `results/` | `v5_numbers.tex`, the machine-written file the dissertation reads its extension numbers from, and the headline results |

Figure code is not included; the figures are in the dissertation.

## Limitations of this copy

This repository documents the work; it is not a package that runs from a clean checkout.

- **No data.** ISIC-DICM-17K, MILK10k, Derm7pt and DFUC2021 are licensed and are obtained
  from their owners (see `LICENSE_AND_DATA.md`). No image, feature array or trained weight
  is distributed here. DFUC2021 is password-protected and the password is not in this code.
- **Colab paths.** The experiments ran in Google Colab on a T4 with the data on Google
  Drive. The paths at the top of each script must be set to a local copy of the data
  before anything runs, and the scripts were condensed from the notebooks after the
  runs; they preserve the settings, rules and order of the experiments but were not
  re-executed in this form.
- **Order matters.** `04` writes the split and the model bundles, `05` writes the cached
  feature arrays, and `06`–`08` read them. `03` writes the per-case concept table that
  `07` uses.
- **Compute.** Training one configuration takes 2–5 minutes on a T4; all twelve
  configuration runs, the five eleven-class runs and the feature passes over 25,000
  images took about forty GPU-hours in total.

Every reported number is fixed by seed 42 and by thresholds set on the calibration split;
the dissertation's Appendix C describes the correspondence between the code and the
results.

Sharia Alam, MSc Data Science, Manchester Metropolitan University, September 2026.
