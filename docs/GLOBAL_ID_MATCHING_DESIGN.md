# Global ID Matching Design

This document describes the current design and intended behavior for **global identity matching** in the ANPR pipeline.

## Goal

Assign a stable `global_id` to the same real-world vehicle across:

- multiple frames in one video
- repeated sightings in later requests
- repeated sightings from the same camera
- repeated sightings from different cameras when features are strong enough

The design prioritizes **plate/OCR evidence** when it is strong and stable.

## Core Principle

Global identity matching is a **best-match reuse** system.

For each new detection, the system:

1. checks whether the same `(camera_id, local_track_id, request_id)` already has an association
2. if yes, refreshes that existing association
3. otherwise searches recent identities from the DB
4. scores each candidate identity
5. reuses the best accepted match or creates a new `global_id`

## Current Match Threshold

A candidate identity is accepted only if:

```text
match_score >= global_id_match_score
```

Otherwise a new identity is created.

Current request-schema default:

```text
global_id_match_score = 0.70
```

## Matching Signals

The current global matcher uses these signals:

- `vehicle_class`
- `vehicle_color`
- `bbox aspect ratio / dimensions`
- `ocr_text` / plate text

## Signal Weights

Plate text is intentionally given the **highest weight**.

Current base weights:

```text
W_CLASS     = 0.20
W_COLOR     = 0.10
W_DIMENSION = 0.15
W_PLATE     = 0.55
```

### Why

Vehicle color and bbox shape are noisy.

Plate text, when available and confident, is the strongest identity signal for ANPR.

So the design now prefers:

- strong plate match first
- then class consistency
- then dimension consistency
- color as a weak helper only

## Adaptive Weight Redistribution

If some signals are absent on both sides, their weights are redistributed proportionally across the available signals.

Example:

- if color is missing on both detections
- its weight is redistributed to the remaining present signals

This prevents missing optional features from unfairly lowering the score.

## Plate Similarity

Plate matching uses **OCR-confusion-aware Levenshtein similarity**.

This helps tolerate OCR confusion such as:

- `0` vs `O`
- `1` vs `I`
- `5` vs `S`
- `C` vs `(` or `{`

So near-identical OCR reads can still match strongly even if one frame has a small OCR artifact.

## OCR Match Threshold For Global Reuse

Accepted OCR text is not automatically trusted for global matching.

For plate-led historical reuse, the fuzzy OCR plate similarity should also satisfy:

```text
plate_similarity >= ocr_match_confidence
```

Current request-schema default:

```text
ocr_match_confidence = 0.85
```

If the current OCR text does not meet this plate-similarity threshold against a stored identity, that candidate should not be reused through OCR-led matching.

## Fuzzy Color Grouping

Color is treated as a supporting signal, not a strict exact-match field.

The matcher normalizes raw colors into broader groups such as:

- `white`, `silver`, `gray` -> `light`
- `black`, `dark_gray`, `charcoal` -> `dark`
- `blue`, `navy`, `cyan` -> `blue`
- `red`, `maroon`, `burgundy` -> `red`

This helps tolerate model confusion such as `silver` vs `white`.

## OCR Confidence Rule

OCR text is only kept in the ANPR result if:

```text
ocr_confidence >= ocr_confidence_threshold
```

Current request-schema default:

```text
ocr_confidence_threshold = 0.5
```

This reduces low-confidence garbage OCR from entering both output JSON and global matching.

## Duplicate Historical IDs Problem

A real vehicle can still end up with two historical IDs if earlier sightings were weak or inconsistent.

Example:

- first weak sighting -> `gid_A`
- later weak sighting -> `gid_B`

This can happen when:

- OCR was absent or poor
- color was unstable
- dimensions were noisy
- camera context was missing or inconsistent

## Future Convergence Rule

If the same real vehicle later appears again with a **strong same-plate match**, the resolver should converge toward one of the existing identities.

### Current tie-break behavior

When multiple recent identities are viable matches and the plate match is strong, prefer:

1. **higher stored plate/OCR confidence**
2. if still tied, **more recent `last_seen_epoch`**

This means future detections should converge to the **latest and strongest historical identity**.

## Important Limitation

This logic does **not merge old duplicate identities in the database**.

So if the DB already contains:

- `gid_A`
- `gid_B`

for the same real car, then:

- future detections should prefer one of them consistently
- but both old rows still remain in the DB

This is **future convergence**, not full historical deduplication.

## If a Vehicle Was Split Earlier

Suppose the same car got two IDs earlier.

Later a new detection arrives with:

- strong OCR text
- high `ocr_confidence`
- same plate

Then the system should:

- search recent identities from the configured lookback window
- identify strong plate matches
- choose the **most reliable existing identity**
- assign that same `global_id` going forward

This gives stable reuse for future sightings.

## Lookback Window

Only recent identities are considered for reuse.

Configured by:

```text
global_track_lookback_seconds
```

The lookup window limits matching to recent history and avoids unbounded scans.

## Camera Context

Global matching quality depends on correct camera context.

The pipeline should pass:

- `camera_id`
- `request_id`
- local `track_id`

through video-frame processing so per-request association refresh works correctly.

## Current Design Summary

The intended behavior is:

- use global tracking after OCR + plate-to-vehicle association
- keep only OCR above threshold
- require fuzzy plate similarity above `ocr_match_confidence` before OCR-led historical reuse
- prioritize plate text over weak visual cues
- compare color using fuzzy color groups instead of strict raw-string equality
- if duplicate historical IDs exist, future strong plate matches should reuse the latest/high-confidence existing identity
- do not create a new `global_id` if a strong recent same-plate match already exists

## Future Upgrade: True Identity Merge

A later improvement can add **historical identity merge**.

That would allow the system to:

- detect that `gid_A` and `gid_B` actually represent the same vehicle
- merge weaker identity into stronger identity
- re-link historical associations

This is not implemented yet.

## Recommended Future Merge Rules

A future merge system should consider:

- exact or near-exact strong plate match
- high OCR confidence on both sides
- same vehicle class
- recency proximity
- same or compatible camera history

and then merge the weaker identity into the stronger one.

## Integration Point

Global matching should happen after:

1. detection
2. local tracking
3. OCR
4. plate-to-vehicle association

This ensures the vehicle record already contains:

- `ocr_text`
- `ocr_confidence`
- `plate_bbox_xyxy`
- `plate_color`

before global identity resolution.

## Output Fields

Each matched detection may contain:

- `global_id`
- `match_score`
- `match_reason`

Example reasons:

- `new_identity`
- `class+plate`
- `class+dimension+plate`
- `class+color+dimension+plate`

## Final Recommendation

For ANPR, the safest identity rule is:

- **plate/OCR should dominate matching when available and strong**
- noisy signals like color should remain secondary
- future sightings should converge to the latest and highest-confidence existing identity for the same plate
- duplicate historical IDs should be tolerated short-term, but a later merge feature should clean them up
