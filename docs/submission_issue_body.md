# EBiM Phase II Policy Submission — Task 3

Official form: https://github.com/EBiM-Benchmark/submissions/issues/new/choose

Select **Phase II Policy Submission** and enter the following values.

## Title

`[Phase II] world model — Task 3`

## Team name

`world model`

## Point of Contact email

Use the Member 1 email from the team's registration record.

## Task

`Task 3 — Assisted Living & Feeding`

## Your assigned Phase II pathway for this task

`Remote — no hands-on session assigned`

## Public GitHub repository URL

`https://github.com/CHANGCHENGGAO/ebim-world-model`

## Pinned commit SHA

`a0b3518fa081aef669a515ed6e9edf9a0f4f55ce`

## Build and run commands

```bash
git clone https://github.com/CHANGCHENGGAO/ebim-world-model.git
cd ebim-world-model
git checkout a0b3518fa081aef669a515ed6e9edf9a0f4f55ce
docker build --build-arg INSTALL_INTERNAL_VISION=0 -t ebim-task3-b2:phase2 .
docker run --rm --network host --ipc host ebim-task3-b2:phase2
```

The image entrypoint waits for the required ROS 2 topics and then launches the
autonomous `full_task3` policy in `real` mode, with Stage 1 and Stage 2 skipped:
Stage 3 drops the whole bowl (beans included) into the recycling bin and Stage 4
places the cup into the sink. It uses the organizer's external 3-D
object-position topic when available; no keyboard, pedal, GELLO or operator
command is accepted during a run.

## Environment and dependencies

- Base image: `ros:jazzy-ros-base` (Ubuntu 24.04, ROS 2 Jazzy).
- Python dependencies are installed by the Dockerfile. CPU wheels are pinned to
  `torch==2.5.1` and `torchvision==0.20.1`; Ultralytics is pinned to `8.3.0`.
- The build host needs internet access to pull the base image and Python/ROS
  packages. The container requires no internet or cloud service at run time.
- The run host must expose the real robot ROS 2 DDS network to the container;
  therefore the command uses `--network host` and `--ipc host`.
- The policy uses the organizer-documented Mobile FR3 Duo topic contract in
  `config/robot_io.json`.

## Hardware assumptions

- Target: organizer Mobile FR3 Duo with two Franka FR3 arms, spine, head RGB
  camera, front/rear LaserScan and swerve base.
- No GPU or cloud service is required. Navigation uses odometry and LaserScan
  feedback. Manipulation begins only when a fresh 3-D object-position topic is
  supplied by the robot-side perception stack.
- The robot begins at the organizer reset pose, with the base software origin at
  that start pose. The measured route in `config/robot_io.json` is +0.85 m,
  -90 degrees, +0.50 m, +1.20 m into the kitchen; after Stage 1, -1.60 m,
  +1.60 m, +0.70 m left, +1.40 m and -90 degrees for Stages 3/4.
- The spine is at its highest safe position during narrow-door transit and is
  lowered to 0.468 m for Stage 1. Arms use submitted safe transit/work poses.
- Objects may vary within organizer reset regions. The policy detects objects
  with its own cameras/YOLO, but the base route and arm motion are
  measured/pre-taught. The dual-LiDAR controller pauses, recentres and retries
  autonomously, giving up only after 15 bounded recovery attempts.
- The organizer must reset robot and task objects between rounds according to
  the official Task 3 procedure.

## Does your policy rely on externally provided object poses?

`Partly — see Notes`

## Did you train on the organizer-released trajectory dataset?

`Our own collected data only`

## What changed since your Phase I submission?

The Phase I technical report (#22) has been replaced by a runnable autonomous
real-robot policy. We added the official ROS 2 contract, measured base route,
dual-LiDAR/odometry safety with autonomous recovery, fail-closed sensor
freshness checks and a deliberately reduced, reliable Task 3 workflow. The
policy skips Stage 1 (table setup) and Stage 2 (feeding); it drops the whole
bowl — beans included — into the recycling bin (Stage 3) and places the cup
into the sink (Stage 4). Object poses come from the robot-side external 3-D
perception stream, never from simulator ground truth. VLA/cloud inference is
not used.

## Optional supplementary links

Leave blank. The final model is contained in the pinned public repository.

## Notes

This issue supersedes Phase I Technical Report issue #22.

The pose answer is `Partly` because task-object observations come from the
robot-side 3-D perception stream, while base and manipulation waypoints are
measured/pre-taught. There is no motion capture, AprilTag input, simulator
ground truth, cloud API or operator input during a run.

Stage 1 (table setup) and Stage 2 (feeding) are intentionally skipped; this is
a fixed scoring strategy, not an operator-selectable run-time branch. Missing
or stale sensors, excessive force or unsafe clearance stop motion and trigger
bounded autonomous recovery rather than requesting human intervention.

## Required checkboxes

Confirm all repository/build statements yourself immediately before filing the
issue, then tick all four submission-requirement boxes and all three
acknowledgement boxes. Organizers evaluate exactly the pinned commit above.
