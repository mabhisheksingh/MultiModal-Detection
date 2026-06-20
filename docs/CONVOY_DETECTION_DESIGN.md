# Convoy Detection Design for Static CCTV URL Sources(Not Implemented)

This document describes a practical design for adding **convoy detection** to the current ANPR pipeline under the following assumptions:

- input source is a **static CCTV camera URL**
- we do **not** receive `lat` / `lon`
- we do **not** receive an upstream `camera_id`
- the current ANPR system already provides:
  - object detection
  - plate OCR
  - local tracking
  - global identity assignment
  - per-frame timestamps within a video/stream session

## Goal

Detect vehicles that appear to travel together in a convoy, using only a **single static camera feed** for the first phase.

For this phase, convoy detection means:

- vehicles move through the same scene
- in the same direction
- with similar apparent speed
- while maintaining a relatively stable gap
- for a sustained time window

This is a **scene-level convoy detector**, not a city-wide or route-wide convoy detector.

## Important Scope Limitation

Because we do not have real `camera_id` or GPS coordinates:

- we cannot compute true inter-camera road distance
- we cannot compute route-level travel speed across sites
- we cannot do cross-location convoy correlation reliably

So this phase is limited to:

- **within-source convoy detection**
- **within-session convoy detection**

## Reuse From Current Pipeline

The existing code already gives us most of the raw signals we need.

### Existing useful signals

From `ANPRService` / `LiveVideoSourceProcessor` output we already have:

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

### Current flow we should extend

Current runtime flow is:

1. video source opened
2. frames sampled and batched
3. inference runs
4. detections merged
5. local tracking assigns `track_id`
6. OCR runs
7. plate-to-vehicle association runs
8. global tracking assigns `global_id`
9. detections are appended to `all_detections`

Convoy detection should be added **after step 9**, using the per-frame resolved detections.

That keeps convoy logic separate from inference logic.

## High-Level Design

### New idea

Build a convoy detector that consumes a list of per-frame vehicle sightings from a single static source and produces convoy candidates.

### Output example

A convoy candidate should contain:

- `source_id`
- `session_id`
- `member_global_ids`
- `member_track_ids`
- `start_ts_ms`
- `end_ts_ms`
- `duration_ms`
- `avg_gap_px`
- `avg_speed_score`
- `avg_alignment_score`
- `convoy_score`
- `reason`

## Source Identity Strategy

Because upstream does not provide `camera_id`, we derive a stable source key from the URL.

### Proposed derived fields

- `source_id`
- `source_name`

### Suggested implementation

For a URL source:

- normalize the URL
- hash it
- store a short deterministic ID

Example:

```text
source_id = sha1(normalized_url)[:12]
```

This allows us to treat the static camera URL as a fixed source identity.

## Detection Data Needed Per Frame

For convoy detection, each vehicle sighting should include:

- `source_id`
- `request_id`
- `frame_id`
- `ts_ms`
- `track_id`
- `global_id`
- `bbox_xyxy`
- `center`
- `name`
- `color`
- `ocr_text`

### Additional derived runtime features

We should compute and store:

- `cx`, `cy` = bbox center
- `bbox_w`
- `bbox_h`
- `apparent_speed_px_per_sec`
- `direction_vector`
- `direction_label`

These can be computed in a convoy service without changing the model outputs.

## Convoy Detection Logic

## Step 1: Filter to vehicle detections only

Ignore plate-only detections.

Use detections where:

- `name` is a vehicle class
- `track_id` is present
- `global_id` is present if available

If `global_id` is missing, fall back temporarily to `track_id` for within-session analysis.

## Step 2: Build per-vehicle trajectories

Group detections by vehicle identity within a single source/session.

Preferred grouping key:

- `global_id`

Fallback key:

- `track_id`

For each trajectory, sort by:

- `frame_id`
- or `ts_ms`

Each trajectory becomes a time series of positions.

## Step 3: Estimate apparent speed

Since we do not have road calibration, use **apparent screen-space speed**.

For consecutive points in a trajectory:

```text
dx = cx_t2 - cx_t1
dy = cy_t2 - cy_t1
distance_px = sqrt(dx^2 + dy^2)
dt_sec = (ts2_ms - ts1_ms) / 1000
speed_px_per_sec = distance_px / dt_sec
```

This is not true km/h.
It is a relative motion estimate suitable for convoy similarity scoring.

## Step 4: Estimate direction

For each trajectory, compute a direction vector using:

- first valid point
- last valid point

```text
dir = normalize([cx_last - cx_first, cy_last - cy_first])
```

Direction similarity between two vehicles can be measured using cosine similarity.

## Step 5: Compute pairwise overlap window

For each pair of vehicles in the same source/session:

- find overlapping time window
- require minimum overlap duration

Example:

- overlap >= 3 seconds
- or overlap >= N tracked frames

If they do not overlap sufficiently, they cannot form a convoy.

## Step 6: Compute pairwise convoy features

For each vehicle pair `(A, B)` inside their overlap window, compute:

### 1. Direction similarity

Use cosine similarity of motion vectors.

Desired behavior:

- close to `1.0` means same direction
- near `0` means unrelated motion
- negative means opposite direction

### 2. Speed similarity

Compare apparent average speed:

```text
speed_similarity = 1 - abs(speedA - speedB) / max(speedA, speedB, epsilon)
```

Clamp result to `[0, 1]`.

### 3. Gap stability

For overlapping timestamps, compute center-to-center pixel gap.

Then measure:

- average gap
- standard deviation of gap

A convoy pair should have:

- gap not too small
- gap not too unstable

Gap stability score can be based on lower variance.

### 4. Temporal persistence

How long do they remain jointly visible and aligned?

Longer overlap strengthens convoy confidence.

### 5. Order consistency

Check whether one vehicle remains consistently ahead/behind the other along the main motion axis.

Frequent order flips reduce confidence.

## Step 7: Pairwise convoy score

Compute a weighted score.

Example:

```text
convoy_pair_score =
    0.35 * direction_similarity +
    0.30 * speed_similarity +
    0.20 * gap_stability_score +
    0.15 * persistence_score
```

Optional penalties:

- subtract if order flips too often
- subtract if overlap is too short

## Step 8: Group formation

Once pairwise scores are computed, build a graph:

- node = vehicle
- edge = pairwise convoy score above threshold

Then derive groups from connected components or clique-like clusters.

For MVP, connected components are enough.

### Example threshold

- edge exists if `pair_score >= 0.72`

### Group score

Use:

- average pair score inside the group
- minimum pair score for stability
- group duration

## Step 9: Output convoy candidates

Each convoy candidate should include:

- source/session identity
- members
- start/end time
- duration
- score
- explanation

Example reason:

```text
same_direction+similar_speed+stable_gap+overlap_8.4s
```

## Recommended MVP Rules

For a first implementation, use these practical conditions.

A convoy pair exists when:

- same source/session
- overlap duration >= `3000 ms`
- direction similarity >= `0.85`
- speed similarity >= `0.70`
- gap coefficient-of-variation <= `0.35`
- pair score >= `0.72`

A convoy group exists when:

- at least 2 vehicles
- all members connected by valid pair edges
- duration >= `3000 ms`

## Data Model Changes

The core ANPR output should remain mostly unchanged.

### New ORM table recommended

Create a `ConvoyEvent` table with fields like:

- `id`
- `source_id`
- `request_id`
- `convoy_id`
- `start_ts_ms`
- `end_ts_ms`
- `duration_ms`
- `member_global_ids_json`
- `member_track_ids_json`
- `convoy_score`
- `reason`
- `created_epoch`

### Optional per-frame sighting table

If we want replayable analytics later, add `VehicleSighting`:

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

For MVP, convoy detection can run in-memory from `all_detections` without first adding DB persistence.

## Service Layer Proposal

### New service

Create:

- `app/services/convoy_detection_service.py`

### Public API

Suggested methods:

- `derive_source_id(source: str) -> str`
- `build_vehicle_trajectories(detections: list[dict]) -> list[...]`
- `detect_convoys(detections: list[dict], source_id: str, request_id: str) -> list[dict]`

### Responsibilities

The service should:

- filter detections
- build trajectories
- compute motion features
- score pairs
- form groups
- return convoy candidates

## Integration Point in Current Pipeline

Best integration point:

- inside `LiveVideoSourceProcessor.process(...)`
- after `all_detections` is complete
- before final response is returned

### Why here

At this point we already have:

- `frame_id`
- `ts_ms`
- local tracks
- global IDs
- full per-video detection list

This avoids touching inference internals.

### Response extension

Add a new top-level field to the video response:

- `convoys`

Example:

```json
{
  "video_path": "...",
  "source_name": "camera_1",
  "total_detections": 128,
  "detections": [],
  "convoys": []
}
```

## API Proposal

No new route is strictly required for MVP.

The existing video processing route can return convoy results inline.

### Later optional route

Add:

- `POST /v1/anpr/detect-convoy`

This could re-run convoy analytics on saved detections or persisted sighting history.

## Code Change Scope

For this static CCTV URL-only MVP, code changes should be moderate.

### New files

- `app/services/convoy_detection_service.py`
- optional repository/model additions if persistence is needed

### Small edits to existing files

- `app/services/video_source_processor.py`
  - call convoy detection after building `all_detections`
  - append `convoys` to response

- `app/services/anpr_service.py`
  - likely no major change needed

- `app/schemas/anpr.py`
  - optional convoy response schema if we want typed responses

## What This MVP Can Detect Reliably

This design can detect:

- two or more vehicles moving together in the same camera scene
- same-direction motion
- similar apparent speed
- stable following gap
- sustained co-movement for a minimum duration

## What This MVP Cannot Detect Reliably

This design cannot reliably detect:

- cross-city convoy behavior
- route-level co-travel across multiple cameras
- GPS distance-window convoy rules
- convoy progression between checkpoints
- legally valid physical speed values

## Phase 2 Future Upgrade Path

If later we add camera metadata, we can extend this design.

### If we later get camera identity

We can:

- persist source metadata explicitly
- correlate convoy behavior per source more cleanly

### If we later get multiple static cameras with known locations

We can:

- compute inter-camera travel times
- estimate segment-level speed
- detect cross-location convoy progression

### If we later get calibration

We can:

- convert pixel motion to real-world speed estimates
- improve gap interpretation

## Recommended Implementation Order

### Phase 1: in-memory static-camera convoy MVP

1. derive `source_id` from URL
2. build trajectories from `all_detections`
3. compute pairwise convoy scores
4. form convoy groups
5. return `convoys` in video response

### Phase 2: persistence

1. store vehicle sightings
2. store convoy events
3. add query endpoints

### Phase 3: multi-source upgrade

1. add explicit source registry
2. correlate across sources
3. add route-level scoring

## Final Recommendation

For the current constraint set, the best design is:

- keep convoy detection as a separate analytics service
- use the static camera URL as a derived `source_id`
- use screen-space speed and stable gap heuristics
- return convoy candidates only for the processed video/session

This gives a usable first version with relatively low disruption to the current ANPR pipeline.

## Real-Time Deployment Q&A

### Q: Can convoy detection run in real time?

Yes, but only if we scope it correctly.

For the current inputs, real-time convoy detection is feasible as:

- per-source convoy detection
- per-session convoy detection
- rolling analysis inside one static camera view

It is not reliable as a true multi-camera route-level convoy detector because we do not have:

- upstream `camera_id`
- GPS or road topology
- calibrated inter-camera travel constraints

So in real time, this feature should be treated as:

- **within-camera convoy detection**
- not **cross-camera convoy progression detection**

### Q: We do not have `camera_id`. How should a real-time system identify the source?

Use the CCTV URL itself as the source identity input.

Recommended approach:

- normalize the URL
- remove unstable auth-only query parameters when possible
- hash the normalized URL
- store a deterministic short `source_id`

Example:

```text
source_id = sha1(normalized_url)[:12]
```

This is good enough for source-level real-time analytics if one URL consistently maps to one physical camera.

### Q: Should convoy detection wait until the whole stream finishes?

No for live deployments.

The offline/video design in this document runs after `all_detections` is complete. For live streams, that should be adapted into a rolling per-source state machine.

Recommended live behavior:

- process each sampled frame result as it arrives
- update active trajectories per vehicle
- update pairwise overlap, speed, direction, and gap statistics
- maintain a rolling window such as last `5-15` seconds
- emit convoy start/update/end events incrementally

So the real-time version should be **incremental**, not strictly post-session.

### Q: Can one machine handle 100-300 live camera URLs?

Not as one monolithic Python process if we expect stable production behavior.

Actual capacity depends on:

- stream resolution
- frame rate
- model cost
- OCR frequency
- batching efficiency
- GPU memory and compute

For planning purposes, `100-300` live feeds should be treated as a distributed workload.

Recommended approach:

- split stream ingestion from inference
- batch frames from multiple cameras for GPU inference
- keep convoy state partitioned by `source_id`
- distribute sources across multiple workers or GPU nodes

### Q: How should live feed processing be structured so latency does not explode?

Use bounded, lossy real-time queues.

Recommended rules:

- keep only the latest frame or a very small queue per source
- drop old frames when downstream is slow
- prefer low latency over full frame retention
- sample aggressively instead of processing every frame

For convoy detection, full video FPS is usually unnecessary.

Practical operating point:

- process approximately `2-5 FPS` per source for convoy analytics
- optionally run detection/tracking at slightly higher rate if needed

Because convoy logic uses sustained co-movement, it generally does not need `25-30 FPS`.

### Q: What happens if the GPU crashes while many streams are running?

If stream ingestion, inference, tracking, and analytics all live in one process, one GPU failure can stop all sources.

That architecture should be avoided.

Recommended production split:

- stream reader / reconnect manager
- inference workers
- analytics workers
- API or event sink

This gives better failure isolation:

- stream readers can continue buffering latest frames
- failed GPU workers can be restarted independently
- analytics workers can tolerate temporary detection gaps
- one worker crash does not have to kill the whole platform

### Q: Should inference and convoy analytics be batched together?

No.

Use different strategies for each layer:

- inference can be batched across many sources for GPU efficiency
- convoy analytics should remain per-source because its state is source-specific

So:

- **global batching for inference**
- **local state for convoy detection**

### Q: What state should a live convoy detector keep per source?

For each `source_id`, keep only rolling analytics state, not unbounded history.

Recommended per-source state:

- active vehicle identities
- recent trajectory points
- smoothed apparent speed
- direction vectors
- active pairwise overlap statistics
- active convoy groups or candidate events
- TTL timestamps for cleanup

Recommended cleanup behavior:

- expire vehicles unseen for `X` seconds
- expire pair state when either member disappears
- close convoy events after inactivity timeout

### Q: Should the convoy logic rely mainly on `global_id` in real time?

Not necessarily.

Inside one live source, `track_id` is often the most immediate and stable signal for active trajectory reasoning. `global_id` is useful as an enrichment layer, but real-time convoy formation inside one source should not depend entirely on perfect global identity assignment.

Practical rule:

- prefer `global_id` when present
- fall back to `track_id` for active within-session convoy state

### Q: What is the recommended real-time architecture for this feature?

Recommended production shape:

1. stream manager opens and monitors live URLs
2. sampled frames are sent to inference workers
3. inference workers return detections with `source_id`, `frame_id`, and `ts_ms`
4. tracking / OCR / association / global identity run
5. convoy detector updates rolling per-source state
6. convoy events are emitted to response, queue, or database

This preserves a clean separation between:

- video ingestion
- model inference
- identity resolution
- convoy analytics
- event persistence

### Q: What is the final production recommendation for real time?

For the current constraints, the safest recommendation is:

- keep convoy detection source-local
- derive `source_id` from normalized URL
- convert post-session logic into rolling-window analytics
- use bounded queues and frame dropping
- batch inference across cameras
- isolate GPU workers from stream readers and analytics workers
- plan horizontal scaling for `100-300` sources

This is the practical path to real-time convoy detection without overpromising city-wide multi-camera convoy intelligence.

## Final Decision Update

The current implementation direction is now updated as follows:

- convoy detection should be treated as a **global-level feature**
- it should not be limited only to one camera-level rolling analysis
- all analytics features should be **enabled by default** unless explicitly disabled later
- upstream input should include `camera_id`, `lat`, `lon`, and zonal coordinates together with the source URL

### Updated meaning for convoy detection

Convoy detection is now expected to operate at a broader correlation level than per-camera anomaly or spatial analytics.

This means the architecture should plan for:

- storage of convoy-relevant events across cameras
- preprocessing and aggregation beyond only in-memory per-source state
- global correlation logic that can use provided `camera_id`, `lat`, and `lon`
- optional use of zonal coordinates attached to the source definition

### Metadata expected from input

For each source, the expected input can include:

- `url`
- `camera_id`
- `lat`
- `lon`
- zonal coordinates in list form

This metadata should be treated as part of the source definition and reused by the convoy pipeline.
