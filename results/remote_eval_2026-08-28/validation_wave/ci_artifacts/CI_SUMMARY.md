# CI Bootstrap Summary -- map_boost TEST (paired, image-level)

seed=42, n_resamples=500, method=paired image-level bootstrap (pycocotools COCOeval, per-dataset RandomState(42))
total wall time: 3374.4s

| Dataset | Metric | Point (new) | CI95 new | Point (baseline) | CI95 baseline | Delta (new-baseline) | CI95 delta | P(delta>0) | Significant |
|---|---|---|---|---|---|---|---|---|---|
| 953 | AP50 agnostic | 0.8419 | [0.8270, 0.8595] | 0.8350 | [0.8180, 0.8528] | +0.0070 | [+0.0027, +0.0135] | 0.994 | YES |
| 953 | mAP50 class-aware | 0.5970 | [0.5751, 0.6239] | 0.5861 | [0.5661, 0.6101] | +0.0108 | [+0.0030, +0.0194] | 0.996 | YES |
| depth | AP50 agnostic | 0.8783 | [0.8541, 0.9009] | 0.8764 | [0.8518, 0.8999] | +0.0018 | [-0.0016, +0.0045] | 0.840 | no |
| depth | mAP50 class-aware | 0.6552 | [0.6211, 0.7020] | 0.6691 | [0.6334, 0.7108] | -0.0139 | [-0.0284, +0.0016] | 0.042 | no |
