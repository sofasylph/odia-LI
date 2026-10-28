# Romanized / Code-mixed Robustness — IndicLID on Bhasha-Abhijnaanam

**Group 4.** Reproduction of the IndicLID romanized language-identification system
([Madhani et al., ACL 2023](https://aclanthology.org/2023.acl-short.71/)), fine-tuning on the
Bhasha-Abhijnaanam benchmark, and transfer evaluation on a native Odia set of YouTube comments.

| | Result |
|---|---|
| Target (paper, IndicLID ensemble) | 80.40% accuracy · 76.44 Odia F1 |
| **Reproduction** — published models, threshold re-selected on validation | **80.46%** accuracy (+0.06) · 72.26 Odia F1 |
| Reproduction — published models, paper's threshold 0.6 | 75.25% accuracy · 67.72 Odia F1 |
| Fine-tuned model, held-out benchmark test | 96.03% accuracy · 96.39 macro F1 · 94.23 Odia F1 |
| Fine-tuned model, native Odia YouTube comments | 52.91% accuracy (= Odia recall) |
| **Translationese gap** (Odia recall, like-for-like) | **41.32 points**, 95% CI [33.36, 47.81] |

---

## Repository layout

```
notebooks/indiclid_ensemble_colab_v2.ipynb   full pipeline: Tracks A, B, C (Colab, T4 GPU)
analysis/robustness_analysis.py              post-hoc analysis (CPU), regenerates results/robustness_analysis.json
results/metrics_all.json                     all metrics written by the notebook's final run
results/robustness_analysis.json             all post-hoc analysis numbers
results/predictions_benchmark_test.jsonl     Track B predictions, held-out benchmark test (5,591 rows)
results/predictions_native_odia.jsonl        Track C predictions, native set (756 rows, comment text removed)
results/Group4_IndicLID_Results.xlsx         results tables; derived metrics are live formulas
models/indiclid_ftr_finetuned.bin            our fine-tuned FastText (Track B/C stage 1)
models/README.md, data/README.md             where to obtain the files not committed here
```

## Experimental design

Three tracks answer three different questions. Their numbers are **not** interchangeable.

| Track | Question | Models | Scored on |
|---|---|---|---|
| **A** | Do the published models reproduce the paper? | Authors' releases, no training | Full romanized benchmark (55,821) |
| **B** | How well can we fine-tune on the benchmark? | Our FastText + fine-tuned BERT | Held-out benchmark test (5,591) |
| **C** | Does the Track B model transfer to real Odia? | Same saved Track B models | Native Odia set (756) |

**Splits.** Bhasha-Abhijnaanam ships as a single test file with no official partition. We
split the 55,821 romanized sentences ourselves — stratified by language, seed 42, 80/10/10:
44,649 / 5,581 / 5,591 (Odia 409 / 51 / 52). Training used the train split only; minority
classes were oversampled to the majority count (78,220 rows). Validation selected the
FastText seed and the ensemble threshold; the test split was scored once.

**Models.** IndicLID is a cascade: a FastText classifier (IndicLID-FTR) labels each sentence,
and inputs where its confidence is ≤ threshold are escalated to a fine-tuned IndicBERT
(IndicLID-BERT). In Track B, FastText was retrained from scratch with the published
architecture (dim 8, char n-grams 3–6, wordNgrams 4) — fasttext cannot resume training a
supervised model — and IndicBERT was fine-tuned with only its final encoder layer, pooler
and classifier unfrozen (7.7M of 278.1M parameters; 1 epoch, lr 2e-5, batch 32).

**Preprocessing.** Track A feeds raw text, matching the official code (`pre_process` is the
identity). Tracks B and C apply the same `normalize()`: collapse whitespace, strip, lowercase.

## Results

### Track A — published-model reproduction (n = 55,821)

| Model | Threshold | % to BERT | Accuracy | Paper | Δ | Odia F1 | Paper | Δ |
|---|---|---|---|---|---|---|---|---|
| IndicLID-FTR alone | — | 0 | 70.70 | 71.49 | −0.79 | 62.57 | 58.26 | +4.31 |
| IndicLID-BERT alone | — | 100 | 79.98 | 80.04 | −0.06 | 71.79 | 76.71 | −4.92 |
| Ensemble | 0.6 | 12.0 | 75.25 | 80.40 | −5.15 | 67.72 | 76.44 | −8.72 |
| Ensemble, threshold selected on validation | 0.95 | — | **80.46**¹ | 80.40 | **+0.06** | 72.26 | 76.44 | −4.18 |

¹ Scored on the 50,240 rows not used to select the threshold. The published target was never consulted.

Both components reproduce within one point, but the ensemble at 0.6 falls short. At 0.6 only
12% of inputs reach BERT; with a *perfect* BERT the ceiling would be 79.89%, so the target is
unreachable at that threshold with the released binary. The paper's Table 9 throughputs imply
~30% escalation at 0.6, i.e. the released FastText produces higher confidences in our
environment than in the authors'. We confirmed the binary is byte-identical to both official
releases and that our routing matches the official code (`pred_score > 0.6`). A fasttext
library-version difference is the remaining suspect; fasttext 0.9.2 does not build on
Python ≥ 3.12, so this is **not directly verified**.

### Track B — fine-tuned model, held-out benchmark test (n = 5,591; Odia n = 52)

| Model | Accuracy | Macro F1 | Odia P | Odia R | Odia F1 | Odia TP / FP / FN |
|---|---|---|---|---|---|---|
| Fine-tuned FastText alone | 95.78 | 95.74 | 100.00 | 96.15 | 98.04 | 50 / 0 / 2 |
| **Fine-tuned ensemble @0.9** | **96.03** | **96.39** | 94.23 | 94.23 | **94.23** | 49 / 3 / 3 |

95% bootstrap CIs: accuracy [95.53, 96.51] · macro F1 [95.67, 97.00] · Odia F1 [88.89, 98.11].
With only 52 Odia test sentences, one sentence moves Odia F1 by 1–2 points.

**Not comparable to the paper's 80.40.** The paper trained on machine-transliterated text and
tested on human-typed romanization; we trained and tested on the same pool. The paper's own
Table 5 shows the same effect: 95.96% when test data matches the training distribution versus
80.40% when it does not.

### Track C — native Odia YouTube comments

The native set has 800 rows; the 756 marked `valid_odia` are scored. **Every scored row is
Odia**, so the model cannot produce a false positive: accuracy equals Odia recall, Odia
precision is trivially 100%, and macro F1 is undefined.

| Slice | Rows | Accuracy |
|---|---|---|
| **All scored rows** | **756** | **52.91** [49.34, 56.48] |
| Romanized | 649 | 57.32 |
| Mixed script | 107 | 26.17 |
| Under 5 words | 177 | 23.16 |
| 5–9 words | 301 | 56.15 |
| 10+ words | 278 | 68.35 |

Odia F1 = 69.20 (precision trivially 100%; not comparable to benchmark Odia F1). 53.0% of rows
were escalated to BERT.

### Translationese gap (Track B − Track C, same model)

| Gap | Value | 95% CI |
|---|---|---|
| **Odia recall, like-for-like (primary)** | **41.32** | [33.36, 47.81] |
| Accuracy, as defined in the brief² | 43.12 | [39.55, 46.74] |
| Odia F1 | 25.03 | [19.23, 30.01] |
| Odia recall, 10+ word sentences only | 28.88 | — (benchmark n = 36) |

² Compares 20-language accuracy with Odia-only accuracy — different quantities.
No bootstrap resample of the primary gap came near zero.

## Why the gap is large

**Length (~12 points).** The benchmark has one Odia test sentence under 5 words; the native
set has 177. Restricting both sides to 10+ words reduces the gap from 41.32 to 28.88.

**Training-data coverage (most of the rest).** Our model learned Odia from 409 encyclopedic
sentences (3,112 distinct words), against 44,240 sentences (205,526 words) for the other 19
languages. Only 34.9% of word tokens in the native comments appear in that Odia training data,
and 29.2% appear only in other languages' training data — versus 57.1% and 13.5% on the
benchmark Odia test. Common Odia words such as *gote* and *jagannath* were seen only under
other language labels.

**Ruled out.** Emoji (stripping them moves FastText recall 44.71 → 44.84); label noise (the
shortest rows — two words or fewer once emoji are removed — were inspected manually and are
short but genuine Odia).

### Published vs fine-tuned on the native set

Because every native row is Odia, recall alone rewards a model that over-predicts Odia. False
alarms were therefore measured on 2,000 held-out non-Odia benchmark sentences, both at full
length and truncated to the native set's length distribution (median 6 words vs 8 native).

| Model | Catches Odia (native) | False "Odia", full-length | False "Odia", short text | Balanced accuracy |
|---|---|---|---|---|
| Published IndicLID-FTR | 93.78 | 0.90 | 2.55 | 95.62 |
| Published ensemble @0.6 | 92.20 | 0.90 | 2.25 | 94.97 |
| Our fine-tuned FastText | 44.71 | 0.00 | 0.05 | 72.33 |

Comparing like with like, the published FastText's Odia recall drops 2.7 points from
benchmark to native (96.48 → 93.78); ours drops 51.4 (96.15 → 44.71). Retraining FastText on the benchmark replaced a model trained on
~100k Odia sentences with one trained on 409: in-domain accuracy rose from 70.70% to 95.78%,
but robustness on real comments fell sharply. Notably, the published **BERT** alone scores only
49.34% on the native set, and raising the published threshold to 0.95 lowers its native recall
to 76.59% — settings and components that excel on the benchmark do not transfer.

## Limitations

- **No real non-Odia negatives.** The native set contains only Odia. Negatives above are
  shortened encyclopedic sentences, not informal comments; the published FastText would need
  a 49% false-"Odia" rate on real non-Odia comments to fall to our balanced accuracy.
- **The 44 `reject` rows are mostly code-mixed Odia**, not other languages, so they cannot
  serve as negatives — and excluding them removes some of the most code-mixed comments.
- **Single annotator** for all 800 native rows; no agreement check.
- **Mixed-script input.** The Track B pipeline is roman-only; the full IndicLID system would
  route low-roman-share text to a native-script model.
- **Our ensemble's false-alarm rate is not measured** — the fine-tuned BERT weights (1.1 GB)
  are not in this repository.
- **Run-to-run variance.** GPU fine-tuning of BERT is not bit-deterministic; an earlier run
  selected threshold 0.6 and scored 96.14%. All figures here come from the final run, which
  produced Track C, so the gap compares one model with itself.

## Methodological notes

- **Target-matched threshold not reported.** A threshold chosen by matching the published
  target on the test set (0.95 → 80.56%) appears in `metrics_all.json` only as
  `track_A_diagnostic_target_matched`. It is evidence of the calibration mismatch, never a result.
- **Validation kept unseen.** An early version trained on train+validation, which made
  validation-based selection meaningless; the final run trains on the train split only.
- **Known artifacts in `metrics_all.json`.** The confidence interval labelled
  `A ensemble (cal.)` belongs to the diagnostic row, not the validation-selected row (binomial
  approximation for that row: ±0.35). The `macro_f1_ci` under `C native Odia` is meaningless for
  a one-class set and should be ignored.
- **Leakage audit.** No `unique_identifier` overlaps across splits. 23 test sentences also occur
  in training, caused by 99 sentences duplicated inside the released dataset.

## Reproducing

**Pipeline (Tracks A–C).** Open `notebooks/indiclid_ensemble_colab_v2.ipynb` in Colab with a T4
GPU and run top to bottom. It downloads the benchmark and published models, writes
`metrics_all.json` and the prediction files, and asks you to upload the native set at Track C.

**Analysis (CPU).** Place the files listed in `data/README.md` and `models/README.md`, then:

```bash
pip install -r requirements.txt
python analysis/robustness_analysis.py
```

This regenerates `results/robustness_analysis.json`. It needs the native set locally to join
comment text, and runs the published BERT on CPU (a few minutes).

## References

- Madhani, Khapra & Kunchukuttan (2023). *Bhasha-Abhijnaanam: Native-script and Romanized
  Language Identification for 22 Indic Languages.* ACL 2023.
  https://aclanthology.org/2023.acl-short.71/
- Benton et al. (2025). *Improving Informally Romanized Language Identification.* EMNLP 2025.
  https://aclanthology.org/2025.emnlp-main.117/
- Dataset: https://huggingface.co/datasets/ai4bharat/Bhasha-Abhijnaanam ·
  Models: https://huggingface.co/ai4bharat/IndicLID-FTR, https://huggingface.co/ai4bharat/IndicLID-BERT
