# Behavioral Pattern Mining Design for Static CCTV URL Sources

This document describes a practical design for adding **behavioral pattern mining** to the current ANPR pipeline under the following constraints:

- input source is a **static CCTV camera URL**
- we do **not** receive upstream `camera_id`
- we do **not** receive upstream `lat` / `lon`
- the current ANPR system already provides:
  - per-frame detections
  - OCR plate text
  - local track IDs
  - global identity assignment
  - timestamps inside a video/stream session

## Goal

Detect unusual or suspicious vehicle behavior patterns such as:

- repeated visits
- lingering / loitering
- repeated appearance near the same static source
- unusual recurrence in manually marked sensitive or low-traffic sources

For the current constraint set, these behaviors are measured **per static source** and **over time**, not over a true GPS map.

## Scope Limitation

True behavior mining often uses:

- camera identity
- site metadata
- latitude / longitude
- geofenced sensitive zones
- traffic baselines across many sites

Because we currently do not receive `camera_id` or GPS:

- we cannot compute true geographic revisit paths
- we cannot do geospatial low-traffic modeling across a region
- we cannot infer movement between real-world coordinates

So phase 1 behavior mining is limited to:

- **source-level repeated visits**
- **source-level lingering**
- **source-level recurrence patterns**
- optional manual source labeling such as `sensitive_source=true`

## Reuse From Current Pipeline

The current ANPR flow already provides the core signals needed.

### Existing useful fields

From resolved detections we already have:

- `frame_id`
- `ts_ms`
- `track_id`
- `global_id`
- `bbox_xyxy`
- `center`
- `name`
- `color`
- `ocr_text`
- `conf`

### Existing pipeline stage to reuse

Behavioral mining should run **after all detections are collected for a source/session**.

This means it should sit on top of:

- detection
- OCR
- plate-to-vehicle association
- global identity assignment

and should not modify inference logic.

## Source Identity Strategy

Since upstream does not provide `camera_id`, derive a deterministic source identity from the static CCTV URL.

### Proposed fields

- `source_id`
- `source_name`

### Suggested approach

- normalize source URL
- hash it
- use a short deterministic identifier

Example:

```text
source_id = sha1(normalized_url)[:12]
```

This lets us treat one static camera URL as one fixed place for behavioral analysis.

## What “Location” Means in This Phase

Without GPS, location is approximated as:

- one static CCTV source
- optionally one scene region inside the source view

This is enough to reason about:

- vehicle revisits to the same source
- lingering in the same source
- repeated recurrence near the same entrance/gate/road view

## Behavioral Use Cases in This Phase

## 1. Repeated visits

The same vehicle identity appears in the same source repeatedly across distinct time windows or sessions.

## 2. Linger / loiter behavior

A vehicle remains visible in the same source longer than expected.

## 3. Repeated dwell near the same scene region

A vehicle repeatedly occupies the same frame area or entry/exit zone.

## 4. Repeated presence in manually tagged sensitive source

If a source is manually marked as sensitive, repeated visits to that source can be flagged.

## 5. Repeated presence in low-traffic source

If a source is manually marked or statistically modeled as low-traffic, repeated recurrence can be flagged as unusual.

## Event Model Needed

Each vehicle event should include:

- `source_id`
- `request_id`
- `frame_id`
- `ts_ms`
- `track_id`
- `global_id`
- `bbox_xyxy`
- `center`
- `vehicle_class`
- `vehicle_color`
- `ocr_text`

### Derived fields

The behavior mining service should derive:

- `cx`, `cy`
- `bbox_w`, `bbox_h`
- `scene_region`
- `entry_zone`
- `exit_zone`
- `episode_duration_ms`

## Core Concept: Vehicle Episodes

Behavior analysis should be based on **episodes**, not raw frames.

An episode is one continuous presence interval for one vehicle identity inside one source/session.

### Episode fields

- `identity_key`
- `source_id`
- `request_id`
- `start_ts_ms`
- `end_ts_ms`
- `duration_ms`
- `entry_zone`
- `exit_zone`
- `avg_center`
- `max_bbox_area`
- `vehicle_class`
- `vehicle_color`
- `plate_text`

Using episodes avoids overcounting a long continuous track as many separate events.

## Scene Regions for Static Cameras

Because the source is static, the frame can be partitioned into consistent regions.

Example:

- left / center / right
- top / middle / bottom
- gate / shoulder / road lane / parking region

This allows behavior rules like:

- vehicle repeatedly appears near the gate region
- vehicle lingers in shoulder region
- vehicle repeatedly pauses in entry zone

## Behavioral Rules

## Step 1: Repeated visit detection

Group episodes by vehicle identity and source.

For each identity/source pair, compute:

- number of distinct visits
- visit timestamps
- average revisit gap
- visit duration distribution

### Distinct visit rule

Two presences count as separate visits if the time gap between episodes exceeds a configurable threshold.

Example:

- `visit_gap_threshold_ms = 300000` for 5 minutes

### Output example

- vehicle `veh_abc123` visited source `cam_x1` 4 times in 24 hours

## Step 2: Linger / loiter detection

For each episode, compute `duration_ms`.

If duration exceeds a threshold for that source or vehicle type, flag it.

### Example thresholds

- moving road feed: linger if visible > `20 sec`
- entry gate feed: linger if visible > `60 sec`

### Optional enhancements

- require low apparent motion during the episode
- require remaining in the same region for most of the duration

## Step 3: Repeated dwell in same region

For each vehicle identity, check whether repeated episodes occur in the same scene region.

Examples:

- repeated stop near gate boundary
- repeated pause near shoulder
- repeated presence near parking edge

This is useful for surveillance-style monitoring.

## Step 4: Sensitive-source recurrence

If a source is manually labeled as sensitive, compute recurrence risk when a vehicle appears repeatedly.

### Example manual metadata

- `is_sensitive_source`
- `site_type`
- `expected_traffic_level`

Without GPS, manual source metadata becomes important for meaningful behavioral alerts.

## Step 5: Low-traffic anomaly detection

For each source, maintain a baseline of typical visit counts by:

- hour of day
- day of week
- vehicle class

Then flag vehicles whose recurrence or linger behavior is unusual for that source context.

### Example

- 3 repeated visits by same vehicle between 2 AM and 4 AM at a normally quiet source

## Behavior Risk Scoring

Create a composite risk score from multiple signals.

Example:

```text
behavior_risk_score =
    0.35 * repeated_visit_score +
    0.25 * linger_score +
    0.20 * region_repeat_score +
    0.20 * sensitive_or_low_traffic_score
```

### Example reason strings

- `repeated_visits_same_source`
- `linger_over_threshold`
- `repeated_presence_same_region`
- `repeat_in_sensitive_source`
- `repeat_in_low_traffic_window`

## Recommended MVP Rules

For the first version, define these simple rules.

### Repeated visit candidate

Flag when:

- same vehicle identity appears in same source
- at least `3` distinct visits
- within configurable rolling window such as `24 hours`

### Linger candidate

Flag when:

- one episode duration exceeds threshold
- and motion is low or region occupancy is stable

### Sensitive-source recurrence candidate

Flag when:

- source is manually marked sensitive
- same identity appears `N` times in rolling window

### Low-traffic recurrence candidate

Flag when:

- source baseline activity is low
- same identity repeats more often than expected

## Persistence Design

Behavior mining becomes much more useful with event persistence.

### Recommended new table: `VehicleSighting`

Suggested fields:

- `id`
- `source_id`
- `request_id`
- `frame_id`
- `ts_ms`
- `track_id`
- `global_id`
- `vehicle_class`
- `vehicle_color`
- `ocr_text`
- `bbox_x1`
- `bbox_y1`
- `bbox_x2`
- `bbox_y2`
- `center_x`
- `center_y`

### Recommended new table: `BehaviorEvent`

Suggested fields:

- `id`
- `source_id`
- `request_id`
- `global_id`
- `behavior_type`
- `start_ts_ms`
- `end_ts_ms`
- `duration_ms`
- `risk_score`
- `reason`
- `metadata_json`
- `created_epoch`

### Optional source metadata table

A manual source registry would improve behavior detection substantially.

Suggested fields:

- `source_id`
- `source_name`
- `is_sensitive_source`
- `expected_traffic_level`
- `notes`

For MVP, behavior detection can run in-memory with optional config-based source labels.

## Service Layer Proposal

Create:

- `app/services/behavioral_pattern_service.py`

### Suggested public methods

- `derive_source_id(source: str) -> str`
- `build_vehicle_episodes(detections: list[dict]) -> list[...]`
- `enrich_detections_with_behavior_state(detections: list[dict], source_id: str, request_id: str) -> list[dict]`

### Responsibilities

The service should:

- build episodes from detections
- aggregate repeat visits
- detect long dwell episodes
- compute risk scores
- enrich each detection with behavioral metadata fields
- keep output compatible with the current response shape

## Integration Point in Current Pipeline

Best integration point:

- `LiveVideoSourceProcessor.process(...)`
- after `all_detections` is complete
- before final response is returned

### Why

At that point we already have:

- all detections for the source/session
- timestamps
- local track IDs
- global IDs

So behavior analytics can remain a separate post-processing layer.

## Response Extension

Do not add a new top-level `behavior_patterns` array for now.

Instead, extend each item in the existing `detections` list with behavioral fields only.

### Detection-level behavioral fields

Add the following fields to each detection:

- `direction_vector`
- `speed_estimate`
- `behavior_state`

Example:

```json
{
  "frame_id": 12,
  "ts_ms": 2400,
  "track_id": "17",
  "name": "car",
  "camera_id": "cam_001",
  "global_id": "gid_101",
  "direction_vector": [0.8, 0.1],
  "speed_estimate": 0.34,
  "behavior_state": {
    "is_repeat_visit": false,
    "is_lingering": false,
    "is_sensitive_zone_presence": true,
    "visit_count": 3,
    "dwell_time_ms": 18000,
    "behavior_label": "repeated_presence",
    "behavior_score": 0.72
  }
}
```

This keeps the current output contract stable while still carrying the behavior analytics signal needed for downstream logic.

## What This MVP Can Detect Reliably

This design can detect:

- repeated visits to the same static source
- long dwell / possible loitering in a static source
- repeated recurrence in the same scene region
- repeated presence in manually marked sensitive sources

## What This MVP Cannot Detect Reliably

This design cannot reliably detect:

- GPS-based site hopping
- region-wide low-traffic modeling across geography
- behavior across unknown physical locations
- geofenced area intrusion without external location metadata

## Upgrade Path

### If later we receive explicit source metadata

We can:

- model sensitive sources more accurately
- define source-specific dwell thresholds
- build traffic baselines by source type

### If later we receive GPS or site coordinates

We can:

- add true location-aware revisit analysis
- detect multi-site repeated behavior
- support geographic sensitive-zone logic

### If later we persist long-term history

We can:

- learn recurring schedules
- detect unusual time-of-day presence
- compute per-vehicle behavioral profiles

## Recommended Implementation Order

### Phase 1

1. derive `source_id` from static CCTV URL
2. build vehicle episodes from `all_detections`
3. detect repeated visits and linger events
4. enrich each detection with `direction_vector`, `speed_estimate`, and `behavior_state`

### Phase 2

1. persist sightings and behavior events
2. add manual source metadata
3. add historical baselines and rolling-window queries

## Final Recommendation

For the current static CCTV URL-only setup, implement behavioral pattern mining as:

- **source-level visit and dwell analytics**
- based on vehicle episodes
- driven by timestamps, recurrence, linger duration, and source-specific rules
- executed after the ANPR video pipeline completes

This gives a practical first version without requiring upstream GPS or camera metadata.

## Real-Time Deployment Q&A

### Q: Can behavioral pattern mining run in real time?

Yes, but not all behavior signals have the same immediacy.

In real time, the easiest signals are:

- active linger or loiter detection
- repeated dwell in the same region during the current session
- repeated re-entry inside a rolling time window

Harder signals require longer history, such as:

- repeated visits over 24 hours
- unusual recurrence at low-traffic hours
- source-specific behavioral baselines

So in production, this should be split into:

- **online behavior detection** for active session alerts
- **historical behavior mining** for long-window recurrence analysis

### Q: We do not have `camera_id`. How should live behavior analytics identify a source?

Use a deterministic `source_id` derived from the normalized static CCTV URL.

Recommended approach:

- normalize the URL
- remove unstable auth-only parameters when possible
- hash the normalized form
- persist a short deterministic identifier

Example:

```text
source_id = sha1(normalized_url)[:12]
```

This is enough for source-level behavior analytics if the URL is stable for the same physical source.

### Q: What behavior patterns are realistic in real time with current inputs?

Reliable phase-1 real-time behaviors are:

- long dwell in one source
- repeated presence in one source during a rolling window
- repeated occupancy of the same scene region
- repeated presence in manually labeled sensitive sources

Not reliable without extra metadata:

- true multi-site path behavior
- city-wide revisit trails
- geofenced sensitive-area analysis
- region-wide traffic anomaly modeling

### Q: Should behavior mining wait until the stream session is complete?

No for live operation.

The offline design in this document analyzes detections after session completion. For production live feeds, behavior logic should run incrementally.

Recommended live behavior:

- update active vehicle episodes continuously
- track dwell duration as the episode grows
- emit alert when linger threshold is crossed
- close and persist episode when the vehicle disappears
- run long-window repeat analysis asynchronously from persisted history

This means live behavior detection should be split into online and historical layers.

### Q: Is in-memory state enough for this feature?

Only for short-window alerts.

In-memory state can support:

- current dwell detection
- current-session revisits
- near-term rolling-window repeat checks

But meaningful behavior mining usually requires persistence for:

- repeated visits across hours or days
- source-level traffic baselines
- low-traffic anomaly scoring
- per-vehicle historical patterns

So persistence is much more important here than for a purely session-local feature.

### Q: What data should be persisted for real-time behavior mining?

At minimum persist:

- completed vehicle episodes
- behavior events
- optional source metadata

Very useful additional persistence:

- rolling per-source traffic counts by hour and day
- per-identity visit history
- region occupancy summaries

Without this, many of the strongest behavior rules remain approximate or unavailable.

### Q: Can one machine handle 100-300 live camera URLs for this feature?

Not safely as one monolithic application if we want stable production behavior.

Recommended architecture:

- stream readers ingest and reconnect live feeds
- inference workers batch frames across many sources
- analytics workers maintain per-source active episode state
- historical miners read persisted episodes and compute long-window patterns

This split is especially important because live dwell alerts and historical recurrence analysis have different compute and storage profiles.

### Q: How should queues behave in a live system?

Use bounded, low-latency queues.

Recommended behavior:

- keep only a very small frame buffer per source
- drop stale frames if inference falls behind
- preserve timestamps even when frames are dropped
- rely on episode logic instead of exact frame continuity

For dwell and revisit analytics, timely state updates matter more than perfect frame retention.

### Q: What if the GPU crashes while streams are active?

Do not couple GPU inference lifetime to all behavioral state.

Recommended separation:

- stream managers continue reconnect and sampling logic
- inference workers restart independently
- active episode state is owned by analytics workers
- completed episodes and behavior events are persisted

This lets the system recover while preserving the historical context needed for repeat-visit mining.

### Q: Should live behavior logic depend mostly on `global_id`?

Use both `global_id` and local session identity carefully.

Recommended rule:

- use `global_id` for cross-session historical aggregation when available
- use `track_id` for immediate in-session dwell and episode continuity

For example, current linger detection inside one live source should not stop working just because cross-session identity resolution is imperfect.

### Q: What is the recommended real-time architecture for this feature?

Recommended production flow:

1. open and monitor live URLs
2. derive `source_id` from normalized URL
3. sample and batch frames for inference
4. run detection, OCR, association, and identity resolution
5. update active per-source vehicle episodes
6. emit online linger or recurrence alerts when thresholds are crossed
7. persist completed episodes and behavior events
8. run asynchronous historical mining for long-window patterns

### Q: What is the role of manual source metadata in real time?

Manual source metadata is very valuable in this feature.

Because we do not have GPS or rich site context, useful behavior scoring often depends on operator-supplied labels such as:

- `is_sensitive_source`
- `expected_traffic_level`
- source type such as gate, road, parking, or checkpoint
- source-specific linger thresholds

This metadata can make alerts much more meaningful even before full geospatial support exists.

### Q: What is the final production recommendation for real time?

For current constraints, the safest recommendation is:

- keep behavior mining source-local in phase 1
- split online dwell alerts from historical recurrence mining
- derive `source_id` from normalized URL
- persist episodes and behavior events early
- use bounded queues and frame dropping
- isolate stream ingestion, inference, analytics, and storage
- rely on manual source metadata to improve alert quality

This is the practical path to real-time behavioral pattern mining without claiming capabilities that require GPS, site topology, or long-term regional history.

## Final Decision Update

The current implementation direction is now updated as follows:

- anomaly and behavioral analytics should be tracked at the **per-camera level**
- all analytics features should be **enabled by default** unless explicitly disabled later
- upstream input should include `camera_id`, `lat`, `lon`, and zonal coordinates together with the source URL

### Updated meaning for behavioral tracking

For this feature, behavior analysis should remain camera-level by default.

This means:

- dwell, revisit, linger, and repeated region occupancy are evaluated per source
- zonal coordinates supplied with the source can be used to define behavioral regions
- source metadata should be reused directly instead of inferring everything only from URL identity

### Performance note for real-time behavior analytics

Real-time behavioral preprocessing is possible, but it adds compute cost.

This is more practical when:

- the number of live cameras is small
- processing is limited to one or a few sources
- behavior rules are narrow and selective

For many simultaneous live CCTV feeds, behavior analytics may need:

- preprocessing and storage support
- asynchronous or partially deferred analysis
- careful control of sampling rate and runtime cost

### Metadata expected from input

For each source, the expected input can include:

- `url`
- `camera_id`
- `lat`
- `lon`
- zonal coordinates in list form

This metadata should be treated as part of the source definition and reused by the behavioral analytics pipeline.
