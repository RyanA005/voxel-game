# neural voxel parkour spec

how the game should work. I wrote this by hand before handing it to Cursor. the entire C implementation came out of a single Composer prompt, but this doc is the source of truth.

small C + raylib 3D voxel parkour with **pure neural physics**. no analytic collision fallback at runtime.

---

## goal

generate a fixed 16×16×16 parkour world on every launch. the player is an AABB that moves, jumps, falls, collides, dies, and reaches a goal. at play time that is driven entirely by a learned MLP.

analytic `physics_step()` stays around as a **teacher** for data recording and benchmarks only.

---

## stack

| component | choice |
|-----------|--------|
| language | C99 |
| rendering | raylib 5.x (CMake FetchContent) |
| ML training | Python 3 + PyTorch |
| inference | pure C MLP (`neural.c`), weights in `models/*.bin` |
| timestep | fixed `dt = 1/60` s everywhere |

### source layout

```
main.c           game loop, CLI modes, rendering
world.c/.h       voxel map, generation, collision queries, goal
physics.c/.h     analytic teacher physics (recording + Tab toggle)
observation.c/.h 9³ patch + feature packing
neural.c/.h      model load, forward pass, neural_physics_step
sim.c/.h         headless record + rollout benchmark
tools/train.py   dataset load, rollout training, export
tools/benchmark.py  one-step + C rollout metrics
common.h         shared types and constants
docs/            blog, progress log, this file
models/          trained weights (see models/README.md)
```

---

## world

```c
#define WORLD_X 16
#define WORLD_Y 16
#define WORLD_Z 16
```

**axes:** x = left/right, y = up/down, z = forward/back. each voxel is a 1×1×1 cube.

**voxel types:**

| value | name | color |
|-------|------|-------|
| 0 | EMPTY | (none) |
| 1 | SOLID | gray |
| 2 | START | blue |
| 3 | GOAL | green |
| 4 | HAZARD | red |

**generation:** random-walk platform path. start platform, 8 intermediate platforms, goal block at the final platform center. regenerated on launch and on win/reset (`R`).

**bounds:** x/z out of map = solid wall. `y < -4` = death (reset player).

---

## player

- AABB: width `0.6`, height `1.8`
- `pos` = body center (not feet)
- spawn: platform top + half height + clearance; bumped up if overlapping solids
- goal: XZ overlap with goal block + (voxel overlap OR feet on goal surface)

---

## controls

| key | action |
|-----|--------|
| W/A/S/D | move (−Z / −X / +Z / +X) |
| Space | jump |
| R | new map |
| Tab | toggle analytic / neural physics |
| N | reload model (fatal if missing) |
| Esc | quit |

---

## physics

### teacher (analytic)

`physics_step(Player *p, InputState input, float dt)` does axis-separated AABB integration with friction, gravity, jump, and per-axis collision resolution.

constants: `MOVE_ACCEL=35`, `MAX_SPEED=6`, `FRICTION=12`, `GRAVITY=-30`, `JUMP_SPEED=10`.

### runtime (neural)

`neural_physics_step(Player *p, InputState input, float dt)`:

1. build observation from local voxel patch + offset + velocity + grounded + keys
2. MLP forward → 7 outputs: `Δx, Δy, Δz, vx', vy', vz', grounded'`
3. apply directly to player state. no collision fix afterward.

**no fallbacks.** missing model = fatal exit. no silent revert to analytic physics.

### model file (`models/*.bin`)

```
magic, version
input_dim, hidden1, hidden2, output_dim
input_mean[], input_std[]
output_mean[], output_std[]
W1, b1, W2, b2, W3, b3
```

C loader reads hidden sizes dynamically (128×128 baseline or 256×256 rollout models both work).

---

## observation and training target

**input (741 for 9³):** `voxels[n³]` (type/4) + offset(3) + vel(3, normalized) + grounded + keys(5)

variable patch sizes **2³ through 9³** supported: center-crop from 9³ recordings at train time; C runtime sets patch from `input_dim` in the model file.

```bash
python tools/train.py --patch 7 --out models/patch_7.bin ...
python tools/patch_sweep.py   # trains + benchmarks 9³ down to 2³ (results → docs/progress-log.md)
python tools/bench_all_models.py   # all models vs analytic (results → docs/progress-log.md)
```

**target (7):** `Δpos(3)` + `next_vel(3)` + `next_grounded`

**dataset:** binary `data/train.bin`, packed `TrainingRecord` (792 bytes), header magic `VPCK`.

record headlessly:

```bash
./build/voxel_parkour.exe --record 500000 --out data/train.bin
```

**dataset v2:** idle stand-still policy (~15% + extra when grounded), 60-frame settle at spawn, sanitized flat-rest targets for idle+grounded rows. train with `--idle-weight 4.0`.

---

## training

```bash
# baseline one-step (128×128)
python tools/train.py --data data/train.bin --out models/model.bin \
  --hidden1 128 --hidden2 128 --no-rollout --epochs 25

# rollout + larger model
python tools/train.py --data data/train.bin --out models/model_rollout.bin \
  --hidden1 256 --hidden2 256 --rollout-steps 8 --rollout-weight 8.0 --epochs 30

# v2: rollout + idle-weight (after re-record)
python tools/train.py --data data/train.bin --out models/model_rollout_v2.bin \
  --hidden1 256 --hidden2 256 --rollout-steps 8 --idle-weight 4.0 --epochs 30
```

**rollout loss:** sample consecutive episode windows; per-step MSE + accumulated Δpos MSE over K frames (teacher-forced observations). val split by episode seed.

single step accuracy is easy to push high. across a rollout it gets much harder and is what actually matters for play feel.

logs include timestamps and ETA per batch/epoch.

---

## CLI modes

| command | purpose |
|---------|---------|
| `./voxel_parkour` | game (default model: `models/patch3/256_rollout8_idle4.bin`) |
| `--record N --out FILE` | headless dataset capture |
| `--bench MODEL` | compare neural vs analytic rollout |

---

## rendering

third-person camera offset `(+8, +7, +8)` from player. cube voxels + wireframes. orange player AABB. HUD: controls, seed, physics mode.

---

## benchmarking

see `docs/progress-log.md` for all the numbers and comparison tables.

```bash
python tools/benchmark.py
./build/voxel_parkour.exe --bench models/model_rollout.bin
```

track one-step RMSE, rollout drift, grounded accuracy, tunnel rate, inference µs. analytic is the oracle row. one step alone tells you very little.

---

## explicit non-goals

textures, chunk meshes, lighting polish, mouse look, enemies, menus, save/load, complex generation, **analytic collision at runtime**.

---

## project log

see `docs/progress-log.md` for the full timeline, expectation changes, and where Cursor tried to screw around.
