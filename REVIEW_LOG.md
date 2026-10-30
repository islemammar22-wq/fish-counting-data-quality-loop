# Review log

Rule applied: an edge fish is labeled only if more than half of it is inside the frame (protocol v2).

## Valid split: corrections applied (`corrections.csv`)

| Image | What was wrong | Decision |
|---|---|---|
| 1281 | Fish cut off at the bottom-right edge had no label | Add label |
| 541 | Tiny sliver at the top edge was labeled | Remove label |
| 1164 | Small piece of head or tail at the edge was labeled | Remove label |

## Valid split: reviewed, left unchanged

| Image | Observation | Decision |
|---|---|---|
| 1185 | Two boxes on one fish (duplicate detection) plus partial fish at the top edge | Leave. Model error, reduced by lowering NMS IoU to 0.5 |
| 200 | Extra prediction on the blurry tail of the same fish | Leave. Ambiguous |

## Train split: worst-error images reviewed, no label changes

| Image | Observation | Decision |
|---|---|---|
| 1082 | Prediction on a thin dark sliver at the right edge | Leave. Edge sliver, fails the more-than-half rule |
| 1420 | Two faint blurry labeled fish the model misses | Leave. Ambiguous, plausible out-of-focus fish |
| 1066 | Blurry S-shaped fish and a smear at the top edge have predictions but no labels | Leave. The correction script would add both boxes, so skipped |
| 1002 | Two crossing fish: extra merged box in the cluster, plus a smudge at the top-left edge | Leave. Duplicate prediction is a model error |

## Outcome
Labels were mostly consistent. After correcting 3 `valid` labels and retraining (round 2), count error on `test` was 0.13 vs 0.16 for round 1, and on `valid` 0.13 vs 0.09. One run each, so I treat the rounds as equivalent. The clearest gain came from lowering the NMS IoU threshold (test MAE 0.30 to 0.16).
