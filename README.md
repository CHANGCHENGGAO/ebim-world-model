# EBiM Phase II Task 3 — B2 release candidate

This directory contains the remote-submission policy container for Task 3. It
connects to an externally started simulator or robot through ROS 2; it does not
package the organizer's scene, robot assets, or evaluation service.

No official score is claimed. Controller-reported stage results are telemetry,
not evaluation evidence. Final scoring must use observations recorded outside
the controller and the organizer-confirmed evaluation geometry.

## Primary design

The default and only enabled policy is `closed_loop`: fresh RGB-D detections,
odometry, measured arm/gripper state, and wrench feedback drive deterministic
motion primitives. Unknown or stale state fails closed. The normal policy does
not read USD/PhysX scene truth and never substitutes fixed object poses.

VLA, LLM planning, Diffusion Policy, and a learned world model are deliberately
excluded from this release candidate. The August 30 trajectory can be evaluated
as an optional single-skill learner after B2 replay works, but it must not replace
the verified controller unless it passes the same postconditions and safety
tests. A generative world model is out of scope for this submission window.

## Runtime components

| Path | Purpose |
|---|---|
| `task3_autonomous.py` | Four-stage closed-loop controller and safe-stop watchdog |
| `vision_callback.py` | Custom-model head/wrist RGB-D perception with camera-to-world TF |
| `b2_contract.py` | Strict sim/real topic, navigation, frame, and limit contract validation |
| `config/robot_io.json` | Single source for the current interface assumptions |
| `b2_evaluator.py` | Score-blind evaluator for external episode observations |
| `entrypoint.sh` | Contract check, topic discovery, perception lifecycle, controller launch |
| `models/README.md` | Offline custom detector requirements |

`bean_counter.py`, `llm_planner.py`, `diffusion_policy.py`, and
`policy_manager.py` are development artifacts and are not copied into the B2
image or used by the default policy.

## Fail-closed contract

`real` is the default competition contract. It is aligned with the organizer's
28-topic MCAP inventory and Franka's Mobile FR3 Duo ROS 2 documentation. The
default workflow is the autonomous reduced route `onsite_bowl_cup`: carry only
the bowl and cup, omit feeding, then perform bean recovery and cleanup at the
final station. `sim` remains selectable for development.

`ALLOW_UNCONFIRMED_IO=1` is for local bench work only. It is not an acceptable
competition setting.

At runtime the controller also requires fresh:

- left/right arm state and measured gripper state;
- left/right wrench feedback;
- odometry;
- organizer-provided target-seat pose;
- world-frame perception output.

A missing or stale required stream latches safe stop, publishes a base stop and
arm hold, terminates the stage, and produces a non-zero process exit. Perception
failure never falls back to the dry-run fixture coordinates.

The default coarse object-pose freshness window is 3.0 seconds
(`DETECTION_MAX_AGE=3.0`). The controller may wait up to 12 seconds to reacquire
a target (`VISION_REACQUIRE_TIMEOUT=12.0`), but immediately before final grasp
approach or pouring it requires an observation no older than 0.8 seconds
(`FINE_DETECTION_MAX_AGE=0.8`). A cached pose never authorizes blind contact
motion; measured-state and force checks remain active.

## Detector contract

Internal perception requires a Task 3 detector at
`models/ebim_task3.pt` or a mounted path selected with
`VISION_MODEL_PATH`. The runtime never downloads a model. Required classes are
documented in `vision_callback.py`; default COCO weights are not sufficient for
beans and competition-specific fixtures.

An external perception node may instead publish the configured object and bean
topics with world-frame coordinates, confidence encoded in each name, and valid
message timestamps. Select it with `VISION_BACKEND=external`.

## Official ROS 2 mapping status

The supplied ROS 2 inventory is reflected in `config/robot_io.json` for real
mode: autonomous base commands use `/swerve_drive_controller/cmd_vel`
(`TwistStamped`), odometry uses `/swerve_drive_controller/odom`, gripper target
commands use `Float32`, and front/rear LiDAR use `/lidar_front/scan` and
`/lidar_rear/scan`. Navigation uses directional scan sectors and the official
TMRv0.2 sensor transforms. Ordinary clearance is 0.40 m; the measured narrow
passage uses 0.05 m nominal and 0.02 m minimum side clearance. A transient
non-contact block triggers autonomous stop/re-observe recovery up to 15 times;
force and hardware safety stops remain latched.

## InnoHub autonomous robustness readiness

The official InnoHub special award runs the submitted Task 3 policy in
simulation with the organizer-side `plus` flag.  This submission accepts no
keyboard, GELLO, or pedal *human input*.  Autonomous base commands may still
be published through the simulator's existing command bridge.

`entrypoint.sh` requires `AUTONOMOUS_ONLY=1` and runs
`autonomy_guard.py` before ROS startup.  It rejects enabled keyboard/GELLO
teleoperation environment switches, controller subscriptions to pedal/keyboard/GELLO
input, and actual ROS/Python teleop helper launches in the entrypoint. Do not set
any `WITH_*_TELEOP` variable in a competition launch.

The detector now measures exposure quality and applies adaptive gamma before
CLAHE.  Nearly black, nearly saturated, or no-contrast frames are discarded;
the existing perception freshness watchdog then stops motion rather than
acting on unreliable observations.

`navigation_safety.py` and `robustness_matrix.py` provide a fail-closed,
interface-independent contract for the three `plus` ground obstacles
(`cable`, `book`, `ball`) and the official 10-point matrix.  The organizer has
not yet published the corresponding topic/prim interface, so these modules do
not guess a topic name.  On release day, connect its adapter to
`plan_safe_path()` and archive the 4-lighting × 4-stage and
3-obstacle × 4-stage results. `robustness_matrix.py --evidence evidence.json`
rejects incomplete matrices, so only complete external observations may be used
to report an award-score result.

## Local checks

For the real-robot, reduced bowl/cup calibration workflow (not an official
four-stage score), follow [the onsite autonomous navigation guide](docs/ONSITE_AUTONOMOUS_NAVIGATION_CN.md).

These checks do not require ROS or the organizer simulator:

```bash
python3 -m unittest discover -s tests -v
python3 task3_autonomous.py --dry-run --stage all --policy closed_loop
bash -n entrypoint.sh
python3 b2_contract.py --config config/robot_io.json --mode sim --allow-unconfirmed
python3 autonomy_guard.py --policy-file task3_autonomous.py --json
```

Before any on-site or Isaac Sim policy launch, run the read-only bridge check.
It creates no publisher and sends no motion command:

```bash
python3 onsite_preflight.py --mode sim
# At the physical test platform (published state/sensor contract only):
python3 onsite_preflight.py --mode real
# Offline regression example:
python3 onsite_preflight.py --snapshot tests/fixtures/task3_sim_bridge_topics.json
```

The check covers the official shared Task 2/Task 3 arm, Robotiq gripper and
`/pedal/state` bridge topics. It intentionally does not declare perception,
target-seat, or Task 3 scoring topics ready; those remain separate policy
readiness requirements.

The evaluator consumes observations exported from rosbag/video annotation and
ignores any score field emitted by the controller:

```bash
python3 b2_evaluator.py \
  --episode /path/to/episode_observations.json \
  --spec /path/to/organizer_confirmed_spec.json
```

`config/evaluation_spec.example.json` is explicitly unconfirmed and must not be
used as official evidence.

## Container

Build the autonomous controller image from this directory. The published real
robot interface contains RGB but no registered depth or camera intrinsics, so
the safe default is `VISION_BACKEND=external`: navigation can leave the reset
pose without waiting for an unavailable RGB-D node. A validated external 3-D
perception adapter is required before any vision-guided grasp/place action can
be attempted:

```bash
docker build --build-arg INSTALL_INTERNAL_VISION=0 -t ebim-task3-b2:phase2 .
```

The optional RGB-D YOLO adapter can be packaged only for a robot contract that
actually supplies synchronized depth and camera calibration; it is not valid
for the currently documented real-robot contract:

```bash
docker build --build-arg INSTALL_INTERNAL_VISION=1 -t ebim-task3-b2:rgbd .
```

Connect the container to the real robot ROS 2 network. The image defaults to
the confirmed autonomous real-robot workflow:

```bash
docker run --rm --network host --ipc host ebim-task3-b2:phase2
```

For Isaac development, override `ROBOT_MODE=sim` and `WORKFLOW=full_task3`.

## Trajectory and interface integration record

The released trajectories and organizer topic inventory were handled as follows:

1. Record its checksum and inspect schema, clocks, frames, action units, joint
   order, gripper convention, camera calibration, and reset semantics.
2. Update `config/robot_io.json` and confirm the documented command/state types.
3. Replay without motion, then run sensor-drop, stale-detection, force-limit,
   empty-spoon, failed-grasp, failed-place, and unknown-bean fault cases.
4. Produce an independent observation export and score it with the confirmed
   geometry. Do not use `STAGE_RESULT` as ground truth.
5. Set `confirmed: true` only after the static contract, route and autonomy
   tests pass; archive checksums with the release package.

## Validation boundary

The policy, route, official topic contract, autonomy guard, detector and offline
tests are packaged. Isaac validates the base/odometry command chain but does not
publish the real dual-LiDAR streams, so final sensor-level and manipulation
validation necessarily occurs on the organizer's Mobile FR3 Duo. Runtime checks
fail closed on missing or stale input and never replace it with hidden truth.
