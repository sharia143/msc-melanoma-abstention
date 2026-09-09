# Results

`v5_numbers.tex` is the file the extension experiments wrote (one LaTeX macro per reported
number, 408 macros); the dissertation reads Sections 4.5 and 4.11-4.13 from it, so every
number in those sections can be traced to this file.

The per-image tables (test-split scores, ulcer decisions, alternative-score grids,
skin-tone audit, seed runs) are on the project's Google Drive and can be provided to the
examiners on request. They are not committed because several contain per-image scores for
licensed images.

Headline numbers (MILK10k test split, 786 lesions, 67 melanomas; C4, seed 42):

| quantity | value |
|---|---|
| AUROC | 0.880 [0.845, 0.911] |
| coverage | 75.6% |
| sensitivity among answered cases | 0.857 (42 of 67 melanomas answered) |
| ECE, before / after temperature | 0.131 / 0.035 |
| thresholds: Gate 1, Gate 2, floor, cut, T | 0.7883, 0.7154, 0.5506, 0.350, 5.068 |
| clinical close-ups of the same lesions refused | 61.6% |
| DFUC2021 ulcers refused / called melanoma | 78.0% / 584 of 9,949 |
| ulcers called melanoma with the wound-aware stage | 5 (scaled) |
| MONET mean AUROC against dermatologist labels (Derm7pt) | 0.632 (published: 0.704) |
| Derm7pt inconsistency (reproduced exactly) | 50 of 305 profiles, 30.3% of lesions, 54.0% of melanomas |
