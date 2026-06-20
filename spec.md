# Neural Voxel Parkour — Spec

A small C + raylib 3D voxel parkour prototype with **pure neural physics** — no analytic collision fallback at runtime.

---

## Goal

Generate a fixed 16×16×16 parkour world on every launch. The player is an AABB character that moves, jumps, falls, collides, dies, and reaches a goal — driven entirely by a learned MLP at play time.

Analytic `physics_step()` remains as a **teacher** for data recording and benchmarks only.

---

## Tech Stack

| Component | Choice |
|-----------|--------|
| Language | C99 |
| Rendering | raylib 5.x (CMake FetchContent) |
| ML training | Python 3 + PyTorch |
| Inference | Pure C MLP (`neural.c`), weights in `models/*.bin` |
| Timestep | Fixed `dt = 1/60` s everywhere |

### Source layout

```
main.c           Game loop, CLI modes, rendering
world.c/.h       Voxel map, generation, collision queries, goal
physics.c/.h     Analytic teacher physics (recording + Tab toggle)
observation.c/.h 9³ patch + feature packing
neural.c/.h      Model load, forward pass, neural_physics_step
sim.c/.h         Headless record + rollout benchmark
tools/train.py   Dataset load, rollout training, export
tools/benchmark.py  One-step + C rollout metrics
common.h         Shared types and constants
```

---

## World

```c
#define WORLD_X 16
#define WORLD_Y 16
#define WORLD_Z 16
```

**Axes:** x = left/right, y = up/down, z = forward/back. Each voxel is a 1×1×1 cube.

**Voxel types:**

| Value | Name | Color |
|-------|------|-------|
| 0 | EMPTY | — |
| 1 | SOLID | gray |
| 2 | START | blue |
| 3 | GOAL | green |
| 4 | HAZARD | red |

**Generation:** random-walk platform path — start platform, 8 intermediate platforms, goal block at final platform center. Regenerated on launch and on win/reset (`R`).

**Bounds:** x/z out of map = solid wall. `y < -4` = death (reset player).

---

## Player

- AABB: width `0.6`, height `1.8`
- `pos` = body center (not feet)
- Spawn: platform top + half height + clearance; bumped up if overlapping solids
- Goal: XZ overlap with goal block + (voxel overlap OR feet on goal surface)

---

## Controls

| Key | Action |
|-----|--------|
| W/A/S/D | Move (−Z / −X / +Z / +X) |
| Space | Jump |
| R | New map |
| Tab | Toggle analytic ↔ neural physics |
| N | Reload model (fatal if missing) |
| Esc | Quit |

---

## Physics

### Teacher (analytic)

`physics_step(Player *p, InputState input, float dt)` — axis-separated AABB integration with friction, gravity, jump, and per-axis collision resolution.

Constants: `MOVE_ACCEL=35`, `MAX_SPEED=6`, `FRICTION=12`, `GRAVITY=-30`, `JUMP_SPEED=10`.

### Runtime (neural)

`neural_physics_step(Player *p, InputState input, float dt)`:

1. Build 741-d observation from local 9³ voxel patch + offset + velocity + grounded + keys
2. MLP forward → 7 outputs: `Δx, Δy, Δz, vx', vy', vz', grounded'`
3. Apply directly to player state

**No fallbacks.** Missing model = fatal exit. No silent revert to analytic physics.

### Model file (`models/*.bin`)

```
magic, version
input_dim, hidden1, hidden2, output_dim
input_mean[], input_std[]
output_mean[], output_std[]
W1, b1, W2, b2, W3, b3
```

C loader reads hidden sizes dynamically (supports 128×128 baseline or 256×256 rollout model).

---

## Observation & training target

**Input (741 for 9³):** `voxels[n³]` (type/4) + offset(3) + vel(3, normalized) + grounded + keys(5)

Variable patch sizes **2³–9³** supported: center-crop from 9³ recordings at train time; C runtime sets patch from `input_dim` in model file.

```bash
python tools/train.py --patch 7 --out models/patch_7.bin ...
python tools/patch_sweep.py   # trains + benchmarks 9³…2³ (results → progress-log.md)
python tools/bench_all_models.py   # all models vs analytic (results → progress-log.md)
```

**Target (7):** `Δpos(3)` + `next_vel(3)` + `next_grounded`

**Dataset:** binary `data/train.bin`, packed `TrainingRecord` (792 bytes), header magic `VPCK`.

Recorded headlessly:

```bash
./build/voxel_parkour.exe --record 500000 --out data/train.bin
```

**Dataset v2:** idle stand-still policy (~15% + extra when grounded), 60-frame settle at spawn, sanitized flat-rest targets for idle+grounded rows. Train with `--idle-weight 4.0`.

---

## Training

```bash
# Baseline one-step (128×128)
python tools/train.py --data data/train.bin --out models/model.bin \
  --hidden1 128 --hidden2 128 --no-rollout --epochs 25

# Rollout + larger model
python tools/train.py --data data/train.bin --out models/model_rollout.bin \
  --hidden1 256 --hidden2 256 --rollout-steps 8 --rollout-weight 8.0 --epochs 30

# v2 — rollout + idle-weight (after re-record)
python tools/train.py --data data/train.bin --out models/model_rollout_v2.bin \
  --hidden1 256 --hidden2 256 --rollout-steps 8 --idle-weight 4.0 --epochs 30
```

**Rollout loss:** sample consecutive episode windows; per-step MSE + accumulated Δpos MSE over K frames (teacher-forced observations). Val split by episode seed.

Logs include timestamps and ETA per batch/epoch.

---

## CLI modes

| Command | Purpose |
|---------|---------|
| `./voxel_parkour` | Game (requires `--model`, default `models/model.bin`) |
| `--record N --out FILE` | Headless dataset capture |
| `--bench MODEL` | Compare neural vs analytic rollout |

---

## Rendering

Third-person camera offset `(+8, +7, +8)` from player. Cube voxels + wireframes. Orange player AABB. HUD: controls, seed, physics mode.

---

## Benchmarking

See `progress-log.md` for all benchmark numbers and model comparison tables.

```bash
python tools/benchmark.py
./build/voxel_parkour.exe --bench models/model_rollout.bin
```

Track one-step RMSE, rollout drift, grounded accuracy, tunnel rate, inference µs.

---

## Explicit non-goals

Textures, chunk meshes, lighting polish, mouse look, enemies, menus, save/load, complex generation, **analytic collision at runtime**.

---

## Project log

See `progress-log.md` for full development history, expectation changes, and benchmark timeline.
