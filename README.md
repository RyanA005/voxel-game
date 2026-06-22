# neural voxel engine

replacing analytic voxel physics with a tiny neural network. C + raylib game, pure C inference at runtime, PyTorch only for training.

full write-up with charts: [docs/blog.html](docs/blog.html) · [ryanhub](https://ryanhub.org/blog/neural-voxel-engine)

## how it works

**the game.** 16×16×16 voxel parkour world. random-walk platform each run. orange AABB player, grey tiles, blue start, green goal. analytic physics exists as a teacher for recording and Tab comparison only. at play time the MLP is the physics engine.

**each frame the model sees:**

- a local voxel patch (default 3×3×3, supports 2³ through 9³)
- sub-voxel position offsets (3)
- velocity (3)
- grounded flag (1)
- WASD + jump inputs (5)
- dt and a few scalars

39 inputs on the default 3³ model. 741 on a full 9³ model.

**the model predicts 7 outputs:** Δposition (3), next velocity (3), next grounded (1). applied directly. no collision fix afterward.

**architecture:** two-layer ReLU MLP. weights exported to a custom `.bin` format and loaded in `neural.c`. forward pass runs in the main thread in pure C with optional AVX2 SIMD. int8/int4/fp16 quant supported if you want smaller files.

**training:** record episodes from the real analytic engine into `data/train.bin`. PyTorch trains with supervised MSE on the 7 outputs, weighted toward position error. the important part is **rollout training**: sample K consecutive frames from one episode, score every step plus accumulated position drift over the window. teacher-forced observations during training, pure closed-loop neural at runtime.

**benchmark rules**:

- pure neural at runtime, no analytic corrections
- fixed `dt = 1/60` everywhere
- analytic physics is the reference oracle
- one-step accuracy is not enough, rollout drift over many frames is what matters

## build it yourself

### requirements

- CMake 3.11+
- C99 compiler (GCC, Clang, or MSVC)
- Python 3 + PyTorch (training and benchmark scripts only)
- raylib fetched automatically by CMake

### 1. build the game

```bash
cmake -S . -B build
cmake --build build
```

binary: `build/voxel_parkour` (or `build/voxel_parkour.exe` on Windows)

### 2. record training data

headless mode runs analytic physics and writes packed samples:

```bash
./build/voxel_parkour --record 500000 --out data/train.bin
```

500k samples takes a few minutes. the dataset is binary with magic header `VPCK`, 792 bytes per record.

### 3. train a model

recommended starting recipe (the idle4 anchor):

```bash
python tools/train.py --data data/train.bin \
  --out models/patch3/256_rollout8_idle4.bin \
  --patch 3 --hidden1 256 --hidden2 256 \
  --rollout-steps 8 --epochs 30 --idle-weight 4
```

~13 minutes on a typical GPU. exports a `.bin` the C loader understands.

other useful commands:

```bash
python tools/train.py --help

# patch size sweep (trains 9³ down to 2³ at 128×128)
python tools/patch_sweep.py

# 3³ architecture experiments
python tools/patch3_experiments.py

# quantize fp32 models to int8/int4/fp16
python tools/quantize_models.py --all-defaults
```

### 4. play

```bash
./build/voxel_parkour --model models/patch3/256_rollout8_idle4.bin
```


| key     | action                   |
| ------- | ------------------------ |
| W/A/S/D | move                     |
| Space   | jump                     |
| R       | new map                  |
| Tab     | toggle analytic / neural |
| N       | reload model             |


missing model file = fatal exit. no silent fallback.

### 5. benchmark

rollout benchmark: 50 episodes × 300 steps, same inputs fed to analytic and neural:

```bash
./build/voxel_parkour --bench models/patch3/256_rollout8_idle4.bin
```

benchmark every model in `models/` and rank them:

```bash
python tools/bench_all_models.py
```

writes results to `docs/progress-log.md`. see [models/README.md](models/README.md) for the weight file layout.

## benchmark results

method: `--bench` rollout, 50 ep × 300 steps, post-AVX2 step times. **pos** = rollout position error vs analytic (lower is better). **gnd mm%** = grounded flag mismatch. **tunnel** = non-grounded overlap steps. **µs** = neural inference per step. analytic is the 0-error oracle at ~0.2 µs.

### reference row


| model    | patch | pos | gnd mm% | tunnel | µs  | KB  |
| -------- | ----- | --- | ------- | ------ | --- | --- |
| analytic | n/a   | 0   | 0       | 0      | 0.2 | 0   |


### models worth knowing about


| model                           | patch | MLP     | train | pos       | gnd mm% | tunnel | µs      | KB   |
| ------------------------------- | ----- | ------- | ----- | --------- | ------- | ------ | ------- | ---- |
| **patch3/256_rollout8_idle4** ★ | 3³    | 256×256 | 13m   | **0.335** | 9.9     | 231    | **9.9** | 304  |
| patch3/512x256_rollout8         | 3³    | 512×256 | 16m   | **0.307** | 13.5    | 686    | 16.5    | 600  |
| patch3/64_rollout4_fast         | 3³    | 64×64   | 5m    | 0.425     | 10.5    | 191    | **1.0** | 28   |
| patch3/256_rollout8_30ep        | 3³    | 256×256 | 11m   | 0.414     | 18.0    | 613    | 7.7     | 304  |
| model_rollout_v2                | 9³    | 256×256 | n/a   | 0.530     | 13.2    | 868    | 31.4    | 1012 |
| model.bin                       | 9³    | 128×128 | n/a   | 0.632     | 14.8    | 798    | 15.5    | 445  |


### idle4 retrain (4³ through 6³, same recipe)


| model                       | patch | pos   | gnd mm% | µs   | KB  |
| --------------------------- | ----- | ----- | ------- | ---- | --- |
| patch_sweep_retrain/patch_4 | 4³    | 0.351 | 10.9    | 9.9  | 342 |
| patch_sweep_retrain/patch_5 | 5³    | 0.374 | 11.3    | 10.3 | 403 |
| patch_sweep_retrain/patch_6 | 6³    | 0.387 | 11.9    | 12.0 | 495 |


### takeaways

- **idle4 anchor** (`patch3/256_rollout8_idle4`) is the sweet spot: ~10 µs, 0.335 pos, 304 KB. best balance we found.
- **512×256 on 3³** is the accuracy ceiling (0.307 pos) at the cost of size and speed.
- **64×64** is the speed floor (~1 µs) with still-decent accuracy (0.425 pos).
- **4³ through 6³ idle4 retrains** cluster right next to the 3³ anchor. patch size 3 is not magic, the train recipe matters more.
- **9³ v2** loses on both speed and accuracy vs the 3³ anchor. more context did not help.
- **128×128 patch sweep models** are under-trained noise. do not read too much into individual sweep numbers.

quantization note: fp16 is ~half file size with +0.008 pos drift. int8 row quant on the anchor is ~80 KB at similar speed. int4 is smaller but much slower to unpack on CPU. full quant tables in [docs/blog.html](docs/blog.html).

## repo layout

```
main.c, world.c, physics.c, observation.c, neural.c, sim.c   # game + inference
docs/spec.md                 # implementation spec
docs/blog.html               # full write-up
docs/progress-log.md         # dev log + auto-updated benchmark tables
models/                      # trained weights (gitignored, see models/README.md)
tools/                       # train.py, benchmarks, quantization
```

questions or ideas: [ralport2005@gmail.com](mailto:ralport2005@gmail.com)