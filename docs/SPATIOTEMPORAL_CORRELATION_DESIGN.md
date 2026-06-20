# Spatiotemporal Correlation Design for Static CCTV URL Sources

This document describes a practical design for adding **spatiotemporal correlation** to the current ANPR pipeline under the following constraints:

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

Detect **repeated co-occurrence patterns** that may signal:

- surveillance behavior
- coordinated movement
- escort behavior
- possible criminal coordination

For the current constraint set, this means identifying vehicles that repeatedly appear together:

- in the same static source
- within the same time windows
- with repeated temporal overlap
- optionally with similar motion behavior in the scene

## Scope Limitation

True spatiotemporal correlation normally uses:

- event timestamp
- camera identity
- camera location or GPS
- physical distance between sightings

Because we do not currently receive `camera_id` or GPS:

- we cannot compute true geographic proximity
- we cannot measure inter-site distance
- we cannot do multi-location route progression reliably

So for phase 1, **space** is approximated as:

- the same static camera source
- the same scene or field of view

This is a valid **source-level spatiotemporal correlation** design, not a city-wide geospatial correlation engine.

## Reuse From Current Pipeline

The current ANPR flow already produces most of the signals needed.

### Existing useful fields

From processed detections we already have:

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

### Existing processing stage we should reuse

Current flow already does:

1. detection
2. local tracking
3. OCR
4. plate-to-vehicle association
5. global identity assignment
6. collection into `all_detections`

Spatiotemporal correlation should run **after `all_detections` is ready**.

## Source Identity Strategy

Since upstream does not provide `camera_id`, derive a deterministic source identity from the CCTV URL.

### Proposed fields

- `source_id`
- `source_name`

### Suggested approach

- normalize source URL
- compute stable short hash

Example:

```text
source_id = sha1(normalized_url)[:12]
```

This lets us treat one static CCTV URL as one fixed place.

## What “Spatial” Means in This Phase

Without GPS, we redefine spatial correlation as one of two levels.

### Level 1: same source

Two vehicles co-occur in the same static source during overlapping time windows.

### Level 2: same scene region

Two vehicles repeatedly co-occur in nearby parts of the same frame over time.

This can be approximated using:

- bbox center proximity
- lane-aligned motion direction
- entry/exit region similarity

## Key Output Types

The service should produce three useful outputs.

### 1. Pairwise co-occurrence summary

For a pair of vehicles:

- how many times they co-appeared
- total overlap duration
- average time gap
- average spatial proximity score
- correlation score

### 2. Coordinated group candidate

A small group of vehicles that repeatedly appear together.

### 3. Repeated association risk flag

Pairs or groups that repeatedly co-occur above thresholds across sessions or repeated windows.

## Detection Event Model

For spatiotemporal correlation, each vehicle event should include:

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

### Derived features

We should additionally derive:

- `cx`, `cy`
- `bbox_w`, `bbox_h`
- `scene_region`
- `motion_direction`
- `entry_zone`
- `exit_zone`

## Scene Zones for Static Camera

Because the source is static, we can divide the frame into reusable logical regions.

### Example zones

- left / center / right
- top / middle / bottom
- entry strip / transit strip / exit strip

These zones allow us to reason about repeated co-occurrence more meaningfully.

Example:

- Vehicle A and Vehicle B repeatedly enter from the left and leave through the right within a small time gap

This is stronger than merely appearing in the same frame.

## Core Correlation Logic

## Step 1: Filter to vehicles only

Ignore plate-only records.

Use:

- vehicle detections with `track_id`
- preferably `global_id`

Fallback to `track_id` inside one session if needed.

## Step 2: Build per-vehicle episodes

Group sightings into **episodes** per vehicle per source/session.

An episode is a continuous interval where the vehicle is tracked in the same scene.

Episode fields:

- `identity_key`
- `source_id`
- `request_id`
- `start_ts_ms`
- `end_ts_ms`
- `duration_ms`
- `entry_zone`
- `exit_zone`
- `avg_center`
- `avg_bbox`
- `motion_direction`

## Step 3: Compare episode overlap between vehicles

For each pair of episodes `(A, B)` in the same source:

- compute temporal overlap
- compute average time separation
- compute scene-region overlap
- compute motion compatibility

### Overlap example

If:

- A visible from `10s` to `18s`
- B visible from `12s` to `19s`

Then overlap = `6s`

## Step 4: Pairwise correlation features

For each pair of overlapping episodes, compute:

### 1. Temporal overlap score

Longer overlap increases confidence.

### 2. Entry-time gap score

If two vehicles enter within a small time gap repeatedly, this is meaningful.

### 3. Exit-time gap score

Consistent exit proximity strengthens association.

### 4. Region proximity score

Compare spatial closeness in the scene.

This can be based on:

- center distance over overlapping timestamps
- same lane/region occupancy
- similar entry and exit zones

### 5. Motion compatibility score

Compare:

- direction similarity
- apparent speed similarity

### 6. Repetition count

If the same pair repeats this pattern multiple times, correlation score increases strongly.

## Step 5: Pairwise correlation score

Example scoring:

```text
pair_correlation_score =
    0.30 * temporal_overlap_score +
    0.20 * entry_gap_score +
    0.15 * exit_gap_score +
    0.20 * region_proximity_score +
    0.15 * motion_compatibility_score
```

Then amplify score if repeated across multiple episodes.

## Step 6: Repeated co-occurrence analysis

A pair should not be flagged only because they overlapped once.

We should aggregate repeated interactions.

### Pair aggregate features

For each pair of identities:

- total shared episodes
- total overlap time
- mean entry gap
- mean exit gap
- mean region proximity
- mean correlation score
- recurrence count across sessions/requests

## Step 7: Coordination risk scoring

Create a higher-level risk score from repeated pairwise evidence.

Example:

```text
coordination_risk_score =
    0.40 * normalized_repeat_count +
    0.30 * average_pair_correlation +
    0.20 * total_shared_duration_score +
    0.10 * consistency_score
```

### Example reasons

- `repeated_same_source_overlap`
- `consistent_entry_gap`
- `similar_motion_pattern`
- `repeated_joint_presence`

## Group Correlation

After pairwise scoring, build a graph:

- node = vehicle identity
- edge = repeated strong co-occurrence

Then derive small candidate groups.

For MVP:

- connected components are enough

This allows detection of:

- 2-vehicle surveillance pairs
- 3-vehicle coordinated groups

## Recommended MVP Rules

For the current static-source-only setup, a pair becomes a correlation candidate if:

- same `source_id`
- overlapping visibility >= `3000 ms`
- entry gap <= `5000 ms`
- exit gap <= `5000 ms`
- motion compatibility >= `0.70`
- repeated in at least `2` episodes or long continuous overlap

A higher-risk pair should require:

- repeated co-occurrence count >= `3`
- stable timing pattern
- high average pair score

## Data Persistence Design

### Recommended new table: `VehicleSighting`

Store individual vehicle sightings or episode summaries.

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

### Recommended new table: `CorrelationEvent`

Store correlated pair/group events.

Suggested fields:

- `id`
- `source_id`
- `request_id`
- `entity_ids_json`
- `start_ts_ms`
- `end_ts_ms`
- `duration_ms`
- `correlation_score`
- `risk_score`
- `reason`
- `created_epoch`

For MVP, in-memory processing is acceptable before persistence is added.

## Service Layer Proposal

Create:

- `app/services/spatiotemporal_correlation_service.py`

### Suggested public methods

- `derive_source_id(source: str) -> str`
- `build_vehicle_episodes(detections: list[dict]) -> list[...]`
- `enrich_detections_with_spatial_state(detections: list[dict], source_id: str, request_id: str) -> list[dict]`

### Responsibilities

The service should:

- create episodes from detections
- compare episodes pairwise
- aggregate repeated associations
- enrich each detection with spatial metadata fields
- keep output compatible with the current response shape

## Integration Point in Current Pipeline

Best integration point:

- `LiveVideoSourceProcessor.process(...)`
- after `all_detections` is collected
- before final video response is returned

### Why

At that point we already have:

- full per-source detection history
- timestamps
- tracking
- global IDs

This avoids changing inference logic.

## Response Extension

Do not add a new top-level `correlations` array for now.

Instead, extend each item in the existing `detections` list with spatial fields only.

### Detection-level spatial fields

Add the following fields to each detection:

- `direction_vector`
- `speed_estimate`
- `spatial_state`

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
  "spatial_state": {
    "active_zone_id": "entry_1",
    "active_zone_type": "entry",
    "is_inside_zone": true,
    "spatial_label": "inside_zone",
    "spatial_score": 0.9
  }
}
```

This keeps the current output contract stable while still carrying the spatial analytics signal needed for downstream logic.

## What This MVP Can Detect Reliably

This design can detect:

- repeated joint presence in the same static source
- repeated short-gap co-appearance
- repeated overlapping movement in the same scene
- possible surveillance-style pairing in one fixed location

## What This MVP Cannot Detect Reliably

This design cannot reliably detect:

- GPS proximity across true geographic locations
- inter-camera spatial distance windows
- route-level progression across city camera network
- real-world proximity outside the source frame

## Upgrade Path

### If later we receive camera identity

We can:

- persist stable source metadata
- aggregate correlations more robustly across requests

### If later we receive GPS

We can:

- add real geospatial distance checks
- correlate across fixed sites
- detect repeated multi-location coordination

### If later we receive explicit zone/site labels

We can:

- assign semantic meanings to sources
- identify correlation near sensitive facilities

## Recommended Implementation Order

### Phase 1

1. derive `source_id` from CCTV URL
2. build episodes from `all_detections`
3. compute pairwise repeated co-occurrence scores
4. enrich each detection with `direction_vector`, `speed_estimate`, and `spatial_state`

### Phase 2

1. persist sightings and correlation events
2. add historical aggregation across requests
3. expose query API

## Final Recommendation

For the current static CCTV URL-only setup, implement spatiotemporal correlation as:

- **source-level repeated co-occurrence analysis**
- driven by timestamps, overlap, region proximity, and motion compatibility
- operating after the ANPR video pipeline completes

This gives a practical first version without requiring GPS or upstream camera metadata.

## Real-Time Deployment Q&A

### Q: Can spatiotemporal correlation run in real time?

Yes, but only as a constrained real-time feature.

With the current inputs, real-time correlation is feasible for:

- repeated co-occurrence inside the same source
- repeated overlap in the same static scene
- short-gap coordinated appearance in one camera view

It is not reliable for true city-wide geospatial correlation because we do not have:

- upstream `camera_id`
- GPS coordinates
- physical distance between sites
- route topology between cameras

So in production, this feature should be described as:

- **source-level spatiotemporal correlation**
- not full **multi-location geospatial correlation**

### Q: We do not have `camera_id`. What should identify a live source?

Use a deterministic `source_id` derived from the normalized CCTV URL.

Recommended approach:

- normalize host, port, path, and stable parameters
- remove unstable auth-only query parameters when possible
- hash the normalized URL
- persist a short deterministic identifier

Example:

```text
source_id = sha1(normalized_url)[:12]
```

This is acceptable if each live URL consistently maps to one physical camera.

### Q: What does “spatial” mean in real time without GPS?

In the current phase, spatial should be interpreted only inside one static source.

Practical meanings are:

- same camera view
- same scene region
- same entry or exit zone
- similar motion corridor or lane-like path

That means the strongest real-time signals are:

- temporal overlap
- repeated entry-time proximity
- repeated exit-time proximity
- repeated same-region co-occurrence
- motion compatibility in the scene

### Q: Should this analysis wait until a stream session is finished?

No for live systems.

The offline form in this document runs after `all_detections` is available. In real time, the logic should be adapted into rolling episode aggregation.

Recommended live behavior:

- update active episodes as detections arrive
- maintain recent co-occurrence statistics for active identity pairs
- close episodes after inactivity timeout
- emit correlation candidates when repeated overlap thresholds are met

So the production version should be **incremental and rolling**, not only end-of-session.

### Q: Can this work across 100-300 live camera URLs?

Yes only with distributed processing and careful state partitioning.

Do not run all feeds, inference, and correlation logic in one monolithic process.

Recommended architecture:

- stream ingestion layer
- batched inference workers
- analytics workers partitioned by `source_id`
- persistent event store for history

This matters more for spatiotemporal correlation because repeated association often depends on history across multiple sessions.

### Q: Is in-memory processing enough for real-time correlation?

Only for a very limited MVP.

In-memory state is enough to detect:

- overlap inside one current session
- repeated presence inside one live process lifetime

It is not enough for durable repeated-correlation analysis across restarts, worker moves, or long time windows.

For useful real-time deployment, persistence is strongly recommended for:

- vehicle episodes or compact sightings
- pair-level aggregate counters
- correlation events

### Q: What state should be maintained per source in real time?

Recommended rolling per-source state:

- active vehicle episodes
- entry and exit zone estimates
- recent motion summaries per identity
- active pair overlap windows
- pair recurrence counters
- last-seen timestamps and TTL cleanup

Recommended persistent state:

- completed episodes
- pairwise repeat counts across sessions
- correlation events with reason and score

### Q: Can we correlate across cameras in real time if URLs are the only source identity?

Not reliably in the strong geospatial sense.

At best, URL-derived identity gives us a stable source key for one camera. It does not tell us:

- which cameras are physically near each other
- whether they observe the same road network
- expected travel time between sources
- whether same-time appearance across sources is meaningful

So cross-camera correlation should remain a future upgrade that depends on explicit source metadata.

### Q: How should live queues be handled for this feature?

Use bounded queues and drop stale frames.

For this feature, low-latency fresh state is more valuable than retaining every frame. Real-time correlation is based on episodes and co-occurrence windows, so aggressive frame retention is not required.

Practical rules:

- maintain very small per-source frame buffers
- sample frames instead of processing every decoded frame
- drop old frames when inference is behind
- preserve timestamps so episode logic remains correct

### Q: What if a GPU worker crashes?

A GPU worker crash should not erase all historical association knowledge.

Recommended separation:

- stream readers keep ingesting and reconnecting
- inference workers restart independently
- analytics state is isolated from GPU process lifetime when possible
- durable counters or completed episodes are persisted

This lets the system recover without losing all repeated-correlation context.

### Q: Should inference batching and correlation batching be designed the same way?

No.

Use:

- batched inference across many cameras for GPU efficiency
- per-source episode and pair state for correlation logic

Inference is throughput-oriented.
Correlation is source-history-oriented.

### Q: What is the recommended real-time architecture for this feature?

Recommended production flow:

1. open and monitor live URLs
2. derive `source_id` from normalized URL
3. sample and batch frames for inference
4. run detection, OCR, association, and identity resolution
5. update per-source vehicle episodes and pair statistics
6. persist completed episodes and correlation events
7. emit real-time correlation alerts or summaries

### Q: What is the final production recommendation for real time?

For current constraints, the safest recommendation is:

- keep correlation source-local in phase 1
- convert post-session analysis into rolling episode-based analytics
- add persistence early for repeated-history use cases
- use bounded queues and frame dropping
- separate stream ingestion, inference, analytics, and storage
- treat cross-camera geographic correlation as a later metadata-driven upgrade

This is the practical way to deliver real-time spatiotemporal correlation without overstating what is possible without camera metadata or GPS.

## Final Decision Update

The current implementation direction is now updated as follows:

- spatiotemporal and spatial analytics should be tracked at the **per-camera level**
- camera-view zones or polygons are part of the intended design
- all analytics features should be **enabled by default** unless explicitly disabled later
- upstream input should include `camera_id`, `lat`, `lon`, and zonal coordinates together with the source URL

### Updated meaning for spatial tracking

For this feature, spatial tracking should be interpreted primarily at the camera-view level.

This means:

- per-camera scene regions remain the main unit of analysis
- zonal coordinates supplied with the source can be used as polygons or operational regions
- live updates should be computed per source using the provided camera metadata and zone definitions

### Metadata expected from input

For each source, the expected input can include:

- `url`
- `camera_id`
- `lat`
- `lon`
- zonal coordinates in list form

This metadata should be treated as part of the source definition and reused by the spatial correlation pipeline.
