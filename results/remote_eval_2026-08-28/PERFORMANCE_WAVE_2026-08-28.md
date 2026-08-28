# Performance wave — 2026-08-28

## Executive result

The locked evaluation supports two different conclusions.  The learned
multi-view/GSP layer produces a statistically supported physical-detection
gain on both datasets.  The mAP branch produces a statistically supported
gain on 953, while its Depth class-aware profile does not generalize and is
not used for production.  The new DINOv2-Large and member-level heads are
validation-only candidates; they have not been used to reopen or select on
the locked test set.

## Locked test lane

All rows below were produced before this validation wave and remain frozen.
The mAP confidence intervals use 500 paired image-level resamples.  The
end-to-end intervals use 5,000 paired tree-level resamples over saved
per-tree summaries; no image, detector, or model inference is performed by
that analysis.

### Detector mAP track

| Test set | Baseline | Locked result | Delta | Paired 95% CI for delta | Supported? |
|---|---:|---:|---:|---:|---|
| 953 agnostic AP50 | 0.834963 | 0.841936 | +0.006973 | [+0.002696, +0.013516] | yes |
| 953 class-aware mAP50 | 0.586107 | 0.596954 | +0.010847 | [+0.002951, +0.019362] | yes |
| Depth agnostic AP50 | 0.8764 | 0.8783 | +0.0018 | [−0.0016, +0.0045] | no |
| Depth class-aware mAP50 | 0.6691 | 0.6552 | −0.0139 | [−0.0284, +0.0016] | no; regression direction |

The exact source is `ci_boot/artifacts/ci_test.json` and
`ci_boot/artifacts/CI_SUMMARY.md`.  The Depth class-aware baseline remains
the production profile.

### End-to-end track

| Dataset | Metric | Baseline | Locked result | Improvement (positive is better) | Paired 95% CI | Supported? |
|---|---|---:|---:|---:|---:|---|
| 953 | physical F1 | 0.804348 | 0.838710 | +0.034362 | [+0.020939, +0.047690] | yes |
| 953 | count MAE | 1.392593 | 1.362963 | +0.029630 | [−0.014815, +0.081481] | no |
| 953 | count ±1 | 0.614815 | 0.637037 | +0.022222 | [0.000000, +0.051852] | no (boundary) |
| Depth | physical F1 | 0.806859 | 0.853408 | +0.046549 | [+0.025733, +0.069025] | yes |
| Depth | count MAE | 0.890909 | 0.772727 | +0.118182 | [−0.036364, +0.263636] | no |
| Depth | count exact | 0.336364 | 0.445455 | +0.109091 | [+0.008864, +0.218182] | yes |
| Depth | count ±1 | 0.809091 | 0.854545 | +0.045455 | [−0.018182, +0.109091] | no |

Locked per-tree sources are
`gsp_artifacts/953/results_test_locked.json`,
`gsp_artifacts/depth/results_test_locked.json`, and the original baseline
summaries named in `ci_boot/artifacts/e2e_paired_test.json`.

The paired end-to-end analysis could not reconstruct class-aware accuracy or
macro-F1 intervals because the legacy baseline stores only aggregate class
results, not per-tree class-correct counts/confusion matrices.  Those two
metrics are therefore reported as point estimates only, never as fabricated
paired CIs.

## Validation-only performance wave

These rows use the frozen TRAIN/VAL linker topology and fit all learned
heads on TRAIN.  Physical detection and count metrics are invariant because
only the class decision is replaced.  They are prospective candidates, not
test claims.

### Class-head candidates

| Dataset | Frozen VAL baseline matched / macro | Candidate | Matched | Macro-F1 | Physical F1 | MAE | ±1 |
|---|---:|---|---:|---:|---:|---:|---:|
| 953 | 0.754204 / 0.601394 | Base member stack (Extra 0.45 + Logistic 0.10, max pool) | 0.760673 | 0.606252 | 0.823216 | 1.252747 | 0.670330 |
| 953 | 0.754204 / 0.601394 | DINOv2-Large Hist, mean pool | 0.764554 | 0.609195 | 0.823216 | 1.252747 | 0.670330 |
| 953 | 0.754204 / 0.601394 | Large + Base + multi-scale residual opinion stack | 0.768435 | 0.610665 | 0.823216 | 1.252747 | 0.670330 |
| 953 | 0.754204 / 0.601394 | Selected Large + Base-logistic stack + B2 logit calibration | 0.768435 | 0.616373 | 0.823216 | 1.252747 | 0.670330 |
| Depth | 0.845652 / 0.680685 | Base member stack (Extra 0.30 + Logistic 0.30, max pool) | 0.854348 | 0.694740 | 0.852641 | 0.931624 | 0.786325 |

The member-stack bootstrap intervals on VAL cross zero, so those earlier
rows are recorded as validation winners without a generalization claim.  The
DINOv2-Large branch was run only on TRAIN/VAL crops and uses 2,048-dimensional
CLS+mean-patch features.  The selected 953 profile uses weights
`large=0.15`, `base_extra=0`, `base_logistic=0.05`, `multiscale=0`, with a
`+0.15` B2 logit bias.  A 5,000-resample paired VAL bootstrap gives matched
accuracy improvement +0.014091, 95% CI [+0.002564, +0.026774],
*p*(positive)=0.988, and macro-F1 improvement +0.014879, 95% CI
[+0.000713, +0.030339], *p*(positive)=0.981.  Both intervals exclude zero;
this remains VAL robustness evidence, not a test claim.

### Pipeline-v2 count compromise

The V2 re-ranked proposal branch improved some physical/class rows but did
not produce a 953 all-rounder satisfying the declared MAE and matched-class
guardrails.  Its strongest relevant rows were:

* 953 best matched: matched 0.757069, F1 0.818947, MAE 1.384615 — rejected
  because MAE is above the 1.35 guardrail.
* 953 best F1: F1 0.823720, matched 0.753927, MAE 1.373626 — rejected
  because both the matched target and MAE guardrail are missed.
* Depth best all-rounder: matched 0.850972, F1 0.844120, MAE 0.811966,
  ±1 0.820513 — retained as a validation diagnostic, not mixed into the
  locked test claim.

The checkpointed count-ensemble follow-up also failed the guardrail: its best
matched row was 0.756443 but MAE was 1.472527.  It is therefore discarded;
the locked count profile remains the reference.

## Ablations and rejected branches

The following branches were executed on TRAIN/VAL and are retained for
auditability:

* detector-only cluster head and Base DINO cluster head;
* member-level Logistic/ExtraTrees heads with mean, max, and top-member
  pooling;
* class-balanced/cluster-balanced member heads;
* RGB multi-scale context branch;
* auxiliary mono-depth / calibrated sensor-depth branch;
* dynamic residual meta-stack;
* DINOv2-Large member head;
* residual MLP member head with an explicit skip connection;
* V2 re-ranked edge/link/count branch.

Auxiliary and dynamic-meta branches did not improve the chosen all-metric
validation profile and are not promoted.  A pure multi-scale 953 row reached
matched 0.761966 but reduced macro-F1 to 0.597531, so it is not the preferred
all-metric candidate.  This is the intended residual/skip policy: retain the
detector anchor and promote a branch only when it improves the declared
downstream objective rather than a single isolated number.

### Additional generalization wave

The following targeted experiments were subsequently run on TRAIN/VAL only:

* tree-level out-of-fold expert stacking, which reached matched 0.754204 and
  macro-F1 0.607873 on 953 and was not better than the selected stack;
* side-aware group aggregation, ordinal logistic heads, DINOv2-Large
  nearest-neighbor/prototype opinions, and a GPU attention head, none of which
  improved the selected 953 profile or the Depth member-stack profile;
* an adaptive Hungarian-versus-GSP policy.  Its TRAIN utility oracle is only
  a diagnostic upper bound (physical F1 0.892258, MAE 0.936951) and is not an
  inference component; learned policies did not produce an all-metric VAL
  improvement;
* rich graph/count features and nonlinear count regressors.  Although their
  internal TRAIN cross-validation error sometimes decreased, the downstream
  VAL physical/count trade-off worsened, so the original count layer remains
  the reference.

These negative results are retained as explicit ablations.  They prevent a
larger model, an apparently lower count-regression error, or a TRAIN oracle
from being mistaken for general pipeline intelligence.  The only promoted
change from this wave is the B2-calibrated 953 class head described above;
physical topology, count targets, and the locked test outputs remain
unchanged.

### Independent-backbone and compromise-selector follow-up

Two targeted follow-ups used only TRAIN-fitted components and frozen VAL
topology.  ConvNeXt-Small, Swin-Tiny, and EfficientNetV2-S were used as
independent timm member opinions.  The strongest standalone 953 opinion
(EfficientNetV2-S ExtraTrees with top-member pooling) reached matched `0.7594`
and macro-F1 `0.6055`, below the selected stack.  A static fusion search
produced a nominal VAL row of matched `0.7697` and macro-F1 `0.6166`, only one
additional correct matched tree over the robust anchor; the best macro row
kept matched at `0.7684` and reached `0.6169`.  Because the increment is
marginal and has no independent paired uncertainty estimate, the timm
opinions are retained as ablations and are not promoted or sent to TEST.

A TRAIN-fitted policy was also tested for choosing between the original Depth
GSP candidate and the V2 geo/count candidate per tree.  The V2-only profile
reduced count MAE from `0.9316` to `0.7607` and raised matched accuracy from
`0.8457` to `0.8495`, but reduced physical F1 from `0.8526` to `0.8341` and
macro-F1 from `0.6807` to `0.6667`.  Learned policies likewise traded away the
declared all-metric objective; the best count-oriented policy had MAE `0.7265`
but physical F1 `0.8421`.  The original Depth GSP remains the
production/reference profile.  This is an explicit skip/compromise layer, not
a hidden test-tuned selection.

## Depth asset verification

The auxiliary asset pass completed 7,044/7,044 mono-depth images (3,992
953 and 3,052 Depth images).  Calibrated sensor-depth reprojection completed
3,052/3,052 files with zero errors.  The measured naive-vs-calibrated shift
was 28.36 px, consistent with the documented approximately 29 px
misalignment.  These assets were used only for validation experiments.

## Reproducibility and protocol guardrails

1. Fit learned components on TRAIN only.
2. Select profiles and opinion weights on VAL only.
3. Keep the original locked test outputs immutable.
4. Run statistical analysis from saved outputs without using test results to
   tune a new profile.
5. Report negative or inconclusive rows explicitly.

The validation harness rejects a `test` split, and the original test runner's
second-opening guard was verified to exit non-zero when invoked again.

## Artifact map

The companion files in this directory contain the locked reports and scripts:

* `GSP_LINKER.md` — GSP/linker method and locked end-to-end artifacts.
* `MAP_BOOST.md` — mAP boost method, test results, and generalization caveat.
* `gsp_artifacts/` — VAL plus locked test GSP JSONs.
* `map_boost_artifacts/` — VAL plus locked test mAP JSONs.
* `scripts/` — test runner and linker implementation copies.
* `ci_artifacts/` — paired mAP and end-to-end bootstrap summaries.
* `validation_wave/` — TRAIN/VAL experiment reports, anchor checks, and
  reproducibility scripts, including the independent-backbone fusion and
  Depth compromise-selector follow-ups.

The full local SHA-256 manifest is generated alongside the staged artifacts.
Large crop arrays and feature matrices remain outside the repository tree;
they are reproducible intermediates rather than publication metadata.

## Latest cross-layer composition (VAL only)

The next layer was deliberately tested after the preceding negative count
ablations. Instead of replacing the original Depth GSP topology, the
experiment crossed the two count-target branches with both topologies, then
applied the already selected class calibration. The strongest point estimate
keeps the original GSP topology and uses the TRAIN-fitted V2 geo Ridge count
target, followed by the predeclared Depth `scale_macro` class calibration:

| Depth VAL metric | Original profile | Cross-layer candidate | Delta |
|---|---:|---:|---:|
| physical F1 | 0,852641 | **0,854225** | +0,001583 |
| count MAE | 0,931624 | **0,914530** | −0,017094 |
| count ±1 | 0,786325 | **0,786325** | 0 |
| matched class accuracy | 0,845652 | **0,850000** | +0,004348 |
| macro-F1 E2E | 0,680685 | **0,689013** | +0,008328 |

This is a genuine all-metric point improvement on the locked validation
profile, and it illustrates the intended layered compromise policy: retain
the GSP recall backbone, borrow the better count target, and let the class
head repair the changed class mix. The paired 5,000-tree bootstrap is still
inconclusive because the validation set has only 117 trees: F1 delta CI
`[-0,006160; +0,009540]`, MAE delta CI `[-0,085470; +0,051282]`, matched
accuracy CI `[-0,011744; +0,021558]`, and macro-F1 CI
`[-0,015425; +0,033033]`. Therefore this is recorded as the strongest
validation candidate, not as a statistically confirmed or test-validated
claim. The full composition and bootstrap are in
`validation_wave/reports/depth_topology_count_class_combo_results_val.json`
and `validation_wave/reports/depth_topology_count_class_bootstrap_val.json`.

A separate head-aware truncation ablation showed why the linker score remains
the primary ranking signal. Member-head confidence increased matched class
accuracy to `0,7718` on 953 and `0,8603` on Depth, but reduced physical F1 by
`0,0064` and `0,0037`, respectively. It is retained as a negative ablation;
the fixed-score selector is not replaced.

## Composition-aware retraining audit

As a final targeted follow-up, a fresh member head was trained on the exact
TRAIN composition (original GSP topology with V2 geo count targets). Its best
VAL row was matched `0,850000` and macro-F1 `0,684983`, versus `0,850000`
and `0,689013` for the existing calibrated head. It therefore adds no gain
and is retained only as a reproducible negative control. The original
composition candidate remains the selected validation point.
