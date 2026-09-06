# 3D Stereo Traffic Perception and Vehicle Trajectory Prediction on KITTI

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/REPO/blob/main/kitti_stereo_trajectory.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/code%20license-MIT-green.svg)](LICENSE)
[![Data: CC BY-NC-SA 3.0](https://img.shields.io/badge/data-KITTI%20CC%20BY--NC--SA%203.0-lightgrey.svg)](http://www.cvlibs.net/datasets/kitti/raw_data.php)

An end-to-end stereo-vision and machine-learning pipeline built on the KITTI raw driving
dataset. Two calibrated cameras on a moving car become metric 3D vehicle trajectories,
speeds, time-to-collision estimates and a learned trajectory predictor — with every claim
measured against real ground truth, and an explicit `NOT AVAILABLE` wherever KITTI provides
none.

```
stereo pair → calibration → disparity → depth → detection → tracking
   → 3D localisation → ego-motion compensation → 3D trajectories
   → speed / acceleration / TTC → behaviour labels → trajectory prediction → Gradio demo
```

One self-contained notebook. Upload to Colab, **Runtime → Run all**. It mounts Drive,
downloads the data itself, caches it, runs a smoke test, then runs the full experiment.
**Total runtime for the published results: 20.6 minutes on a Colab T4.**

---

## Results

All numbers below are from a single full run over 7 KITTI drives (1,104 frames, 5,332 track
observations, 256 tracks). Test drives were untouched until final evaluation.

| Component | Metric | Result | Ground truth used |
|---|---|---|---|
| Stereo depth (dense) | MAE | **1.569 m** | Velodyne LiDAR |
| Stereo depth (dense) | RMSE | 4.772 m | Velodyne LiDAR |
| Stereo depth (dense) | bad-3px | 6.09 % | Velodyne LiDAR |
| Stereo depth (dense) | δ < 1.25 | 0.944 | Velodyne LiDAR |
| Learned stereo (RAFT, cross-task) | MAE | 2.193 m | Velodyne LiDAR |
| Vehicle detection | AP@0.5 | **0.807** | KITTI tracklets |
| Vehicle detection | recall@0.5 | 0.970 | KITTI tracklets |
| Detection (all classes) | mAP@0.5 | 0.903 | KITTI tracklets |
| Object 3D localisation | depth MAE | **2.361 m** | KITTI tracklets |
| Object 3D localisation | lateral MAE | 0.737 m | KITTI tracklets |
| Object 3D localisation | success rate | 90.2 % | KITTI tracklets |
| Speed estimation | MAE | **1.350 m/s** (4.86 km/h) | OXTS ego velocity |
| Speed estimation | RMSE | 2.570 m/s | OXTS ego velocity |
| Ego-motion compensation | apparent speed of parked cars, before | 32.45 km/h | OXTS (truth = 0) |
| Ego-motion compensation | apparent speed of parked cars, after | **5.62 km/h** | OXTS (truth = 0) |
| Behaviour classification | macro F1 | 0.545 | *derived pseudo-labels* |
| Behaviour classification | accuracy | 0.658 | *derived pseudo-labels* |
| Trajectory prediction | ADE, constant velocity | **1.260 m** | KITTI-derived trajectories |
| Trajectory prediction | ADE, GRU | 1.858 m | KITTI-derived trajectories |
| Trajectory prediction | FDE, constant velocity | **2.214 m** | KITTI-derived trajectories |
| Trajectory prediction | FDE, GRU | 4.063 m | KITTI-derived trajectories |
| Tracking | IDF1 / MOTA | `NOT AVAILABLE` | needs the KITTI Tracking benchmark |
| Motion anomaly detection | tracks flagged | 13 of 142 | no ground truth exists |
| Motion anomaly detection | method agreement | 85.2 % | IsolationForest vs autoencoder |

Depth error by distance band, which is the single most important table in the project:

| True depth | Stereo depth MAE | Pixels/frame |
|---|---|---|
| 0–10 m | **0.215 m** | ~4,192 |
| 10–20 m | 0.906 m | ~4,641 |
| 20–40 m | 2.614 m | ~2,354 |
| 40–80 m | 7.102 m | ~1,158 |

---

## Key findings

**1. Ego-motion compensation is a prerequisite, not a refinement.**
With the ego-vehicle driving at ~37 km/h, *parked* cars appear to move at **32.45 km/h** in
the raw camera frame. After compensating with the OXTS GPS/IMU poses, their measured speed
drops to **5.62 km/h** against a true value of zero. Skip this step and every speed,
behaviour label and trajectory in the pipeline is wrong by roughly the ego-vehicle's speed.

**2. Stereo depth error grows quadratically with range — measured, not asserted.**
Differentiating `Z = fB/d` gives `|∂Z/∂d| = Z²/(fB)`, predicting that one pixel of disparity
error costs 6.5 cm at 5 m and 16.7 m at 80 m. The measured MAE goes 0.215 → 7.102 m across
the distance bands: a **33× degradation** that closely tracks the geometry. This is why the
project reports depth by band rather than as a single average, and why everything downstream
(speed, TTC, trajectories) is trustworthy near the car and increasingly not at range.

**3. The classical method beat the learned one on depth.**
StereoSGBM: **1.563 m** MAE, 6.25 % bad-3px. Pretrained RAFT-small applied to the rectified
pair: 2.193 m MAE, 16.34 % bad-3px — on the identical frames under the identical LiDAR
protocol. RAFT produces near-total coverage (98 % vs 69 %) but is substantially less
accurate, which is expected: it is an *optical-flow* model used cross-task, not a dedicated
stereo network. Reported as measured rather than quietly dropped.

**4. Constant velocity beat every learned model over a 1-second horizon. ← the honest negative result**

| Model | ADE (m) | FDE (m) | RMSE (m) |
|---|---|---|---|
| **Constant velocity** | **1.260** | **2.214** | 4.302 |
| Transformer | 1.352 | 2.627 | 2.044 |
| Random forest | 1.577 | 2.860 | 2.648 |
| GRU (main model) | 1.858 | 4.063 | 2.604 |

The GRU is **47.4 % worse** than doing no learning at all. This is a real finding, not a bug:
over 1.04 s at urban speeds vehicles genuinely do travel at near-constant velocity, and 1,087
training windows from four drives is not enough to learn anything better. Crucially, the GRU
predicts *absolute* displacements rather than residuals on top of the baseline — a residual
parameterisation would have made "beating" constant velocity almost automatic and the
comparison meaningless.

The per-horizon breakdown is more interesting than the headline:

| Horizon | Constant velocity | Transformer | GRU |
|---|---|---|---|
| t+0.1 s | **0.132** | 0.185 | 0.361 |
| t+0.5 s | 1.248 | **1.165** | 1.584 |
| t+0.6 s | 1.438 | **1.404** | 1.857 |
| t+0.7 s | 1.629 | **1.556** | 2.154 |
| t+1.0 s | **2.214** | 2.627 | 4.063 |

The Transformer **beats constant velocity in the 0.5–0.7 s band**, where a vehicle's
manoeuvre has begun to diverge from pure extrapolation but noise has not yet dominated. It
loses at both ends. That crossover is where the learning signal actually lives, and it is the
strongest argument in this project for longer prediction horizons (3–5 s) and more data.

**5. A linear model cannot do the behaviour task; a non-linear one can.**
Logistic regression collapses to macro F1 **0.078**; a random forest on the same features
reaches **0.545** and a GRU 0.380. The kinematic decision boundaries (turning requires *both*
high yaw rate *and* speed above a threshold) are conjunctive and non-linear, so the linear
baseline earns its place by failing informatively.

---

## Classical computer vision vs machine learning

This project is frequently mislabelled as "a machine-learning project". It is mostly not.
Most of the geometric backbone is deterministic multi-view geometry with **zero learned
parameters**:

| Stage | Family | Learned parameters? |
|---|---|---|
| Camera intrinsics/extrinsics, projection model | Classical geometry | No |
| Epipolar geometry, rectification | Classical geometry | No |
| StereoSGBM disparity, `Z = fB/d` triangulation | Classical CV | No |
| 3D back-projection, point cloud | Classical geometry | No |
| Ego-motion compensation from GPS/IMU | Classical geometry / sensor fusion | No |
| Kalman filter + Hungarian assignment tracking | Classical estimation | No |
| Speed, acceleration, TTC (finite differences) | Classical kinematics | No |
| RAFT disparity (optional comparison) | Deep learning | Pretrained only |
| YOLOv8n detection | Deep learning | Pretrained only |
| Behaviour classification | Classical ML / DL | **Trained here** |
| Trajectory prediction (GRU / Transformer) | Deep learning | **Trained here — primary ML task** |
| Motion anomaly detection | Unsupervised ML | **Fitted here** |

Only three components are trained in this notebook. The detector and the learned stereo model
use off-the-shelf weights, and **no detector training is claimed anywhere**.

---

## Ground truth: what is real, and what is not

The project distinguishes four categories throughout, and labels every result accordingly:

| Category | Source | Used for |
|---|---|---|
| **Ground truth** | Velodyne HDL-64E scans (in the raw archives) | Dense depth/disparity error |
| **Ground truth** | KITTI tracklet 3D boxes (`tracklet_labels.xml`) | Detection AP, 3D localisation error |
| **Ground truth** | OXTS RT3003 GPS/IMU velocity | Ego-motion, speed validation |
| **Derived labels** | Kinematic thresholds in `CONFIG` | Behaviour classification (pseudo-labels) |
| **Model predictions** | YOLO, GRU, Transformer, RAFT | Everything predicted |
| **Heuristic estimates** | TTC, anomaly scores | Ranking attention only |

**Nothing is fabricated.** Where KITTI provides no labels — MOT identity metrics, anomaly
labels, per-vehicle speed truth — the notebook prints `NOT AVAILABLE` with the reason and the
benchmark that *would* be required.

### Validation protocols worth reading before quoting the numbers

- **Speed** is validated indirectly. KITTI gives no ground-truth speed for *other* vehicles,
  so the pipeline selects tracks it judges stationary in the world frame and checks their
  camera-relative speed against the OXTS ego velocity. This exercises the whole chain
  (disparity → depth → 3D position → differentiation) against real truth, but it measures
  static objects, so **1.350 m/s is a lower bound** on moving-vehicle error.
- **Detection precision (0.343) is understated by design.** Ground-truth boxes are filtered
  to ≤60 m, ≥25 px tall, occlusion < 2 and truncation < 2, leaving 99 GT vehicles against 280
  detections. Most "false positives" are real cars that the filter excluded. Recall (0.970)
  and AP (0.807) are the meaningful figures here. This is *not* the official KITTI protocol
  and must not be compared against leaderboard numbers.
- **3D localisation has a known −2.0 m bias.** Tracklet ground truth marks the 3D box
  *centre*; the stereo estimate marks the visible *near surface*, roughly half a vehicle
  length closer. The bias is reported rather than removed; the de-biased MAE is 1.933 m.

---

## No data leakage

Frames within one KITTI drive are 10 Hz samples of the same vehicles on the same street —
consecutive frames are near-duplicates. **Randomly splitting frames would put a car's position
at *t* in training and at *t*+0.1 s in test**, inflating every metric. The split is therefore
by *drive*:

| Split | Drives | Windows | Tracks |
|---|---|---|---|
| Train | 0009, 0011, 0018, 0057 | 1,087 | 63 |
| Validation | 0059 | 310 | 15 |
| Test | 0013, 0014 | 234 | 11 |

Enforced at runtime by assertions that no track and no drive appears in two splits. The
feature scaler is fitted on training windows only; validation selects the training epoch; the
test drives are touched exactly once, at final evaluation.

---

## Automated data acquisition

No manual download, no upload, no login. The notebook fetches only what it needs from KITTI's
public object store:

- **Selective mode** (small jobs) reads the remote ZIP's central directory and pulls
  individual members via HTTP range requests — only `image_02`, `image_03`, `oxts`, capped
  LiDAR, and timestamps, restricted to the frame budget.
- **Bulk mode** takes over above ~250 members, because each range request costs ~1 s and a
  single streamed archive is far quicker. The published run downloaded ~8 GB of archives in
  ~8 minutes and extracted only the 160 frames per drive it needed.
- **Never downloaded:** the greyscale cameras, the unsynchronised `*_extract` archives, or any
  unrelated KITTI benchmark. LiDAR is capped at 24 scans/drive since it is only used as the
  depth reference.
- **Cached in Drive** and skipped entirely on re-runs; temporary archives go to local scratch,
  never to Drive.
- **Staged to local disk** before processing. Reading thousands of small PNGs through Drive's
  FUSE mount was measured at ~2.5 s/frame against ~0.5 s of actual computation; bulk-copying
  first removes that overhead.

If the host is unreachable, Section 5 stops with an explicit message and a manual-placement
path rather than failing obscurely later.

---

## How to run

1. Open the notebook in Colab and select a GPU runtime (`Runtime → Change runtime type → T4`).
   CPU works but is much slower; the notebook detects and adapts.
2. Leave `CONFIG["debug_mode"] = True` and **Runtime → Run all**. This smoke-tests every
   component on 24 frames/drive in ~10 minutes. Expect several `NOT AVAILABLE` results —
   24 frames cannot feed a trajectory model, and the notebook says so rather than pretending.
3. Set `CONFIG["debug_mode"] = False` in the single configuration cell (Section 4) and run
   again for the full experiment (~20 minutes on a T4, ~8 GB downloaded once).

There is exactly **one** cell to edit. Everything is controlled from `CONFIG`:

```python
CONFIG = {
    "debug_mode": True,          # ← the only switch you normally touch
    "random_seed": 42,
    "use_google_drive": True,    # cache survives runtime resets
    "auto_download": True,
    "train_drives": ["0009", "0011", "0018", "0057"],
    "val_drives":   ["0059"],
    "test_drives":  ["0013", "0014"],
    "max_frames": 160,           # per drive
    "obs_len": 10, "pred_len": 10,   # 1.0 s → 1.0 s at 10 Hz
    ...
}
```

A **smoke test runs before the full pipeline** and asserts on every component (environment,
dataset, calibration, stereo, detection, tracking, 3D localisation, ego-motion, schema), so a
broken stage surfaces in seconds rather than after twenty minutes of processing.

Per-drive results are cached under a hash of the settings that affect them, so an interrupted
run resumes without reprocessing completed drives.

---

## Notebook structure

34 sections, grouped:

| Sections | Content |
|---|---|
| 1–7 | Overview, research questions, environment, `CONFIG`, automated acquisition, verification, dataset exploration |
| 8–11 | Camera calibration, the pinhole model, epipolar geometry, rectification verification |
| 12–13 | Classical StereoSGBM depth + LiDAR evaluation; optional learned (RAFT) comparison |
| 14–15 | Pretrained detection with tracklet evaluation; ByteTrack-style SORT |
| 16–18 | 3D localisation, ego-motion compensation, trajectory construction and smoothing |
| 19–22 | Speed, acceleration, time-to-collision, behaviour classification |
| 23–28 | Trajectory dataset, sequence-level split, baselines, GRU/Transformer, ADE/FDE evaluation, anomaly detection |
| 29–34 | Integrated pipeline, animation, results dashboard, Gradio demo, limitations, future work, conclusions |

The geometry sections are written as explanations, not equation dumps — why `f_x` is measured
in pixels, why the homogeneous division *is* the projection, and why depth being inversely
proportional to disparity dooms far-field accuracy.

### Implementation notes

- **Calibration is never hard-coded.** `f_x = 721.54 px` and `B = 0.5327 m` are parsed from
  `calib_cam_to_cam.txt` and validated by six sanity checks, including that the two rectified
  cameras are vertically aligned to within 3.05 mm.
- **Tracking is implemented from scratch** (~150 lines): a constant-velocity Kalman filter per
  track, optimal Hungarian IoU assignment, and ByteTrack's two-stage association that recovers
  partly occluded vehicles from low-confidence detections. No extra dependency to break.
- **3D localisation** uses the bounding box's bottom-centre (the road contact point, stable
  under viewpoint change) with a robust median taken in *disparity* space over the lower
  central patch, since `Z = fB/d` is non-linear and averaging depths ≠ averaging disparities.
- **Trajectories are agent-centric**: translated to the last observed position and rotated so
  heading points along +x, so the model learns manoeuvres rather than map locations.
- **Smoothing is Savitzky-Golay** (7 frames, order 2) rather than a moving average, which
  would flatten genuine braking events and bias the behaviour labels toward "cruising".

---

## Limitations and observed failure modes

Honest reporting includes the things that went wrong in this run.

**The anomaly detector mostly found perception failures, not unusual driving.** The
highest-scoring tracks show `speed_max` of 108–186 m/s and accelerations beyond ±350 m/s² —
physically impossible values produced by depth noise and ID switches, not by dangerous
manoeuvres. This is exactly the caveat the notebook states up front, now confirmed
empirically: it is **motion anomaly detection**, and a useful practical use of it here is as a
*data-quality filter* rather than a driving-behaviour monitor.

**Some TTC values are artefacts.** The worked example reports a 59.5 m/s closing rate at
4.2 m range. Real closing rates in these clips are ~10 m/s; the rest is differentiated depth
noise at close range on a partly occluded box. The gating (positive closing rate, lateral
alignment, valid depth, sustained ≥3 frames) removes 81 % of observations but cannot remove
this class of error. TTC here is an **estimated perception metric for ranking attention, not a
safety-certified function**.

**Behaviour labels are partly circular.** They are derived from the pipeline's own kinematics
via documented thresholds, so a classifier given speed and acceleration could in principle
re-derive them. Two mitigations: labels use a *centred* (non-causal) window while features are
strictly *causal* and unsmoothed, and evaluation is on held-out drives. The `cruising` class
still collapses to F1 = 0.00 — it is the residual "none of the above" category and is
intrinsically ill-defined.

**Other known limits.** COCO-pretrained detection has no `Van` class and misses distant or
heavily occluded objects. SORT has no appearance model, so similar cars crossing can swap IDs.
OXTS drifts in urban canyons. Only single-agent context is used — no lanes, no signals, no
interaction between vehicles, which is precisely the information needed for the turning and
yielding cases where constant velocity fails. Scope is one recording date (2011-09-26),
daytime, dry, urban Karlsruhe: nothing here should be assumed to transfer to night, rain or
highway speeds.

---

## Future work

1. Fine-tune the detector on KITTI to recover `Van` and improve small/occluded recall.
2. Use a dedicated stereo network (RAFT-Stereo, CREStereo) instead of a repurposed flow model,
   scored on the official KITTI Stereo 2015 protocol.
3. Appearance-based tracking (DeepSORT / BoT-SORT) and the KITTI Tracking benchmark for real
   IDF1/MOTA.
4. **Scale the trajectory dataset and extend the horizon to 3–5 s.** The Transformer's win in
   the 0.5–0.7 s band is the concrete evidence that learning has something to offer here once
   extrapolation stops being sufficient.
5. Add social/interaction and map context to the predictor.
6. Probabilistic multi-modal prediction (best-of-K / NLL), which matches the genuine ambiguity
   of driving far better than a single deterministic path.
7. A 3D IMM/Kalman filter on object state with uncertainty propagated into TTC.

---

## Reproducibility

| | |
|---|---|
| Environment | Google Colab, Tesla T4 |
| Python / PyTorch | 3.13.15 / 2.11.0+cu128 |
| NumPy / pandas / OpenCV | 2.1.3 / 2.2.3 / 5.0.0 |
| Seed | 42 (`random`, `numpy`, `torch`, `PYTHONHASHSEED`) |
| Frames | 160 per drive (1,104 total), native 10 Hz (measured Δt = 0.1035 s) |
| Runtime | 20.6 minutes end to end, including download |

Every run writes `results/summary_<mode>.json` (config, package versions, calibration, model
hyper-parameters, all metrics) and `results/results_dashboard.csv`.

Dependencies install automatically: `ultralytics`, `gradio`, `remotezip` on top of Colab's
stock scientific stack. No paid services and no API keys.

---

## Repository contents

```
kitti_stereo_trajectory.ipynb   the complete pipeline (66 cells) with outputs from the full run
README.md                       this file
```

The committed notebook retains its execution outputs, so all figures, tables and metrics are
viewable on GitHub without running anything.

---

## License and citation

Code in this repository: MIT.

**Data: the KITTI raw dataset is licensed CC BY-NC-SA 3.0 (non-commercial).** No KITTI data is
redistributed here; the notebook downloads it from the official source at runtime. If you use
this work, cite the dataset authors:

```bibtex
@article{Geiger2013IJRR,
  author  = {Andreas Geiger and Philip Lenz and Christoph Stiller and Raquel Urtasun},
  title   = {Vision meets Robotics: The KITTI Dataset},
  journal = {International Journal of Robotics Research (IJRR)},
  year    = {2013}
}
```

## Acknowledgements

KITTI by Karlsruhe Institute of Technology and Toyota Technological Institute at Chicago.
Pretrained detection weights from [Ultralytics](https://github.com/ultralytics/ultralytics);
RAFT weights from `torchvision`. The tracker is an original implementation of the
[SORT](https://arxiv.org/abs/1602.00763) and [ByteTrack](https://arxiv.org/abs/2110.06864)
ideas.
