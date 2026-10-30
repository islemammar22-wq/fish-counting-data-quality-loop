# Annotation protocol (v2)

Dataset: Fish-Counting (Roboflow, CC BY 4.0), one class. Goal: label every fish that should be counted, in a way two annotators would agree on.

## What to label
- One box per individual fish.
- Box tightly around the visible body, no extra margin.

## Edge cases (decide once, apply everywhere)
- **Cut off by the image border:** label only if **more than half** of the fish is inside the frame. Tiny slivers or a small piece of head or tail at the edge are **not** labeled.
- **Overlapping or crossing fish:** one box per fish. Boxes may overlap. Never draw one merged box around two fish.
- **Head or tail of a fish whose body is visible:** it is the same fish. No extra box.
- **Blurry or out of focus:** label only if the elongated body shape is recognizable at normal viewing size. Skip faint smudges. If unsure, skip and note it as ambiguous in the review log.
- **Debris, reflections, shadows, tank wall:** never label.

## Quality control
1. Run the model and compare predicted and labeled counts per image (`evaluate.py`).
2. Review the images with the largest differences (`review.py`: green = label, red = prediction).
3. Classify each case: missing label, wrong label, model miss, duplicate prediction, or ambiguous.
4. Fix only clear cases. Do not change a label just to improve the score. Ambiguous cases stay as they are and are logged.
5. Log every decision in `REVIEW_LOG.md`, apply fixes with `apply_corrections.py`, retrain, and re-measure on the same labels.

## Prediction settings for counting
Confidence 0.25 and NMS IoU 0.5. The default NMS IoU of 0.7 produced duplicate boxes on single fish and a higher count error.

## Change log
- v1: initial rules.
- v2: made the edge rule explicit (tiny slivers are not labeled), added rules for crossing fish, tails and heads, and blurry fish, based on the review of the largest validation and training errors.
