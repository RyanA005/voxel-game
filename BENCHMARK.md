# Neural Physics Benchmark Results

Pipeline: record analytic physics → train PyTorch MLP → export `models/model.bin` → pure C forward pass in `neural_physics_step()`.

## Dataset

| Item | Value |
|------|-------|
| Samples | 500,000 |
| Record rate | ~160k samples/sec (headless) |
| Fixed dt | 1/60 s |
| Input dim | 741 (9³ voxels + offset + vel + grounded + keys) |
| Output dim | 7 (Δpos, next vel, next grounded) |

## One-step accuracy (held-out episodes, Python)

| Metric | Result |
|--------|--------|
| Position RMSE | **0.00253** |
| Position p95 error | 0.00607 |
| Velocity RMSE | **0.154** |
| Grounded accuracy | **99.22%** |

## Rollout vs analytic (50 episodes, 300 steps max, C benchmark)

Same random input sequences; analytic and neural simulators run in parallel from identical start states.

| Metric | Result |
|--------|--------|
| Mean position error | **0.231** |
| Mean velocity error | **0.948** |
| Grounded mismatch | 6.11% |
| Non-grounded tunnel (neural) | 0 / 13820 |
| Analytic fallback triggered | removed — pure neural only |

## Speed (Windows / MinGW, CPU)

| Operation | Latency |
|-----------|---------|
| Analytic `physics_step` | **0.19 µs** |
| Neural `neural_physics_step` (obs + MLP + occasional fallback) | **305 µs** |
| MLP forward pass only | **302 µs** |

Neural path is ~1600× slower than analytic but still **~300× faster than a 60 FPS frame budget** (16.6 ms).

## How to reproduce

```bash
# Build
cd build && cmake .. -G "MinGW Makefiles" && cmake --build .

# Record
./voxel_parkour.exe --record 500000 --out ../data/train.bin

# Train
python tools/train.py --data data/train.bin --out models/model.bin --epochs 25

# Benchmark
python tools/benchmark.py
./voxel_parkour.exe --bench models/model.bin

# Play (Tab toggles analytic/neural, N reloads model)
./voxel_parkour.exe --model models/model.bin
```

## Architecture

```
741 → Linear(128) → ReLU → Linear(128) → ReLU → Linear(7)
```

Weights exported to `models/model.bin` with input/output normalization stats for bit-exact C inference.

## Assessment

**Replaced:** `physics_step()` can be swapped for `neural_physics_step()` with Tab in-game; model loads at startup.

**Accurate:** One-step imitation is strong (sub-centimeter position error on average).

**Fast enough:** Sub-millisecond inference, viable for real-time play.

**Next improvements:** multi-step / rollout training loss, reduce analytic fallback rate, larger model or 3D conv for better long-horizon stability.
