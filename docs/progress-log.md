# Progress Log — Neural Voxel Parkour

Chronological record of prompts, decisions, expectation changes, model performance, and benchmarks.

---

## Phase 1 — Prototype engine

**Prompt:** Implement `spec.md` as simply as possible.

**Delivered:**

- Single `main.c` voxel parkour (later split into modules)
- 16³ world, random-walk platforms, AABB analytic physics
- raylib rendering, CMake + FetchContent
- `physics_step()` isolated; observation patch + before/after seam for future ML

**Status:** Playable prototype — move, jump, collide, fall, goal.

---

## Phase 2 — Neural physics plan

**Prompt:** Plan data collection → training → inference → benchmarking.

**Decisions:**

- Fixed `dt = 1/60`
- 9³ voxel patch, 741-d input, 7-d output (Δpos, vel, grounded)
- Pure C MLP inference (no ONNX)
- Headless `--record`, PyTorch train, export `model.bin`
- Benchmark: one-step RMSE + rollout vs analytic + speed

**Expectation:** Replace `physics_step` with `neural_physics_step`; analytic stays as teacher/oracle.

---

## Phase 3 — Full pipeline implementation

**Delivered:**

- `world.c`, `physics.c`, `observation.c`, `neural.c`, `sim.c`, `tools/train.py`, `tools/benchmark.py`
- 200k → 500k sample recording
- 128×128 MLP, ~741→128→128→7
- In-game Tab toggle, `--bench` mode

### Model: `models/model.bin` (128×128, one-step loss, with collision fallback)


| Metric            | Value              |
| ----------------- | ------------------ |
| Val pos RMSE      | 0.00433            |
| Val vel RMSE      | 0.26               |
| Grounded acc      | 99.0%              |
| Rollout pos error | 0.23               |
| Analytic fallback | **52.7%** of steps |
| Analytic step     | ~0.2 µs            |
| Neural step       | ~305 µs            |


**Note:** Fallback (revert to analytic on overlap) made rollouts look better but violated "pure neural" intent.

---

## Phase 4 — No fallbacks, gameplay fixes, retrain

**Prompt:** No fallbacks anywhere; fail if model missing; fix spawn-inside-block; fix goal logic; retrain.

**Expectation change:** **Zero** analytic fallback at runtime. Missing model = fatal exit, not silent analytic mode.

**Code changes:**

- Removed collision fallback from `neural_physics_step`
- `neural_require_model()` — game/bench/reload exit on load failure
- Spawn: `platform_top + half_height + 0.05`, bump loop if colliding
- Goal: AABB overlap + feet-on-surface detection (not center-point inside voxel)

### Model: `models/model.bin` (128×128, one-step, pure neural)


| Metric                       | Value                             |
| ---------------------------- | --------------------------------- |
| Val pos RMSE                 | 0.00315                           |
| Val vel RMSE                 | 0.16                              |
| Grounded acc                 | 99.2%                             |
| Rollout pos error            | **0.68** (worse without fallback) |
| Neural tunnel (non-grounded) | 941 / 12475                       |
| Analytic step                | ~0.2 µs                           |
| Neural step                  | ~64 µs (Release build)            |


**Gameplay:** Goal touch works when walking onto green block. Spawn stable. Sink/float/jitter reported under pure neural.

---

## Phase 5 — Stability discussion

**Prompt:** How to improve sink / float / jitter?

**Options discussed:**


| Approach                            | Decision                                                        |
| ----------------------------------- | --------------------------------------------------------------- |
| A — Neural vel + analytic collision | **Rejected** — project goal is to abandon physics_step entirely |
| B — Runtime floor snap / rules      | Band-aid only                                                   |
| C — Rollout training loss           | **Approved**                                                    |
| D — Bigger model + more training    | **Approved**                                                    |


**Rationale:** Stability must come from better learned dynamics, not reintroducing analytic collision.

---

## Phase 6 — Rollout training + docs

**Prompt:** Implement rollout training; progress/ETA logging; update spec; start progress log; train new model.

**Delivered:**

- `tools/train.py` rewrite:
  - Episode-aware rollout loss (K-step Δpos accumulation, teacher-forced obs)
  - Configurable `--hidden1/2`, `--rollout-steps`, `--rollout-weight`
  - Timestamp + ETA on batch and epoch logs
- `spec.md` rewritten to match current architecture
- This file

**Training started:** `models/model_rollout.bin` — 256×256, 8-step rollout, 30 epochs, 500k samples.

### Model: `models/model_rollout.bin` (256×256, 8-step rollout loss) — **COMPLETE**

| Metric | Baseline 128×128 | Rollout 256×256 |
|--------|------------------|-----------------|
| Val pos RMSE (train) | 0.00315 | **0.00281** |
| Val rollout pos 8-step (train) | — | **0.00458** |
| C rollout pos error | 0.68 | **0.46** |
| C rollout vel error | 2.62 | **1.84** |
| Grounded mismatch | 18.6% | **10.9%** |
| Neural tunnel | 941 | **357** |
| Step µs | ~66 | ~153 |
| × analytic | ~330× | ~765× |

*Analytic oracle (reference): 0 rollout error, ~0.2 µs/step (1×).*

Training time: **10m 56s** on CPU. Log: `models/train_rollout.log`. Best checkpoint epoch 29.

**Try it:** `./build/voxel_parkour.exe --model models/model_rollout.bin`

---

## Phase 7 — Idle dataset + flat-rest targets (sink fix)

**Prompt:** Document rollout; add standing-still data; clean grounded teacher targets (no micro-bounce).

**Hypothesis:** Teacher idle frames have tiny `dy`/`vy` from gravity→collision resolve. Network learns downward drift without enough zero-motion examples.

**Changes:** idle recording policy, 60-frame spawn settle, `sanitize_grounded_idle_targets()`, `--idle-weight 4.0`.

**Retrain:** `models/model_rollout_v2.bin` on re-recorded `data/train.bin`.

### Model: `models/model_rollout_v2.bin` — **COMPLETE** (13m 39s)

| Metric | Rollout v1 | Rollout v2 (idle dataset) |
|--------|------------|---------------------------|
| Val pos RMSE | 0.00281 | **0.00217** |
| Val rollout pos 8-step | 0.00458 | **0.00272** |
| Grounded acc | 99.2% | **99.8%** |
| C rollout pos error | 0.46 | **0.42** |
| C rollout vel error | 1.84 | **1.08** |
| Grounded mismatch | 10.9% | **8.7%** |
| Neural tunnel | 357 | 420 |

*Analytic oracle (reference): 0 rollout error, ~0.2 µs/step (1×).*

**Try it:** `./build/voxel_parkour.exe --model models/model_rollout_v2.bin`

**Remaining issues (user report):** slight phasing through blocks; difficulty descending. v2 overcorrected sink at cost of fall/edge behavior.

---

## Phase 8 — Patch size sweep (9³ → 2³)

**Prompt:** Train models for patch sizes 9³ down to 2³; benchmark whether smaller patches lose performance (faster train/inference).

**Delivered:**

- Dynamic patch via `obs_set_patch_n()` — inferred from model `input_dim` (`n³ + 12`)
- `tools/patch_sweep.py` — trains 9→2, benchmarks each
- `train.py`: `--patch N` center-crops from 9³ recordings; v1/v2 dataset support; memory-efficient loading
- Models: `models/patch_sweep/patch_N.bin`

**Config:** 128×128 MLP, 4-step rollout, 15 epochs, 200k samples per patch (~17 min total on CPU).

### Results 

| Patch | Input dim | Rollout pos ↓ | Rollout vel | Grounded mm% | Tunnel | Step µs | × analytic | Forward µs |
|-------|-----------|---------------|-------------|--------------|--------|---------|------------|------------|
| *(analytic teacher)* | — | 0 | 0 | 0 | 0 | **~0.2** | **1×** | — |
| 9³ | 741 | 0.6753 | 2.083 | 14.7 | 437/13679 | 67.4 | ~337× | 65.0 |
| 8³ | 524 | 1.6862 | 3.510 | 26.8 | 521/13931 | 50.1 | ~251× | 46.3 |
| 7³ | 355 | 0.6903 | 2.128 | 21.8 | 314/14107 | 38.6 | ~193× | 33.6 |
| 6³ | 228 | 1.2078 | 2.064 | 10.8 | 128/14026 | 22.5 | ~113× | 20.7 |
| 5³ | 137 | 0.6270 | 1.661 | 9.5 | 276/13957 | 15.1 | ~76× | 16.2 |
| 4³ | 76 | 0.5384 | 1.617 | 11.9 | 111/14433 | 12.5 | ~63× | 12.3 |
| 3³ | 39 | 1.3130 | 2.476 | 12.2 | 283/14044 | 9.7 | ~49× | 9.0 |
| 2³ | 20 | 0.3920 | 1.403 | 9.1 | 316/13701 | 8.8 | ~44× | 9.4 |

**Why so variable?** Not monotonic with patch size — likely a mix of:

1. **Under-trained sweep models** — 128×128, 15 epochs, 200k samples vs full 256×256 v2 trainer; high variance between runs.
2. **Information vs capacity tradeoff** — smaller patches see less context (more tunneling risk) but also fewer input dims (easier to fit with same 128×128 width).
3. **Center-crop blind spots** — edges/ledges may fall outside a 2³–4³ window; some sizes accidentally align better with platform geometry in the benchmark episodes.
4. **Compounding rollout error** — C benchmark uses pure neural multi-step; a model that is slightly worse one-step can look better or worse at 400 steps depending on error sign (drift into vs away from geometry).

**Practical takeaway:** 4³–5³ looks like a reasonable speed/quality tradeoff; 9³ remains safest for max context. 2³ winning on rollout pos is suspicious — treat as benchmark noise until retrained at 256×256.

---

## Phase 9 — Dataset v3 + closed-loop training (in progress)

**Prompt:** Test dataset v3 (edge-walk + fall policies, lighter sanitization) + closed-loop rollout in `train.py`; export `model_rollout_v3.bin`.

**Hypothesis:** v2 fixed sink but hurt falling (heavy idle + aggressive sanitization). Phasing persists because teacher-forced rollout never trains on self-predicted observations.

**Dataset v3 (format v2, 804-byte records):**

- Added `pos_before[3]` to `TrainingRecord` for closed-loop state init
- Policies: walk-to-edge (~15%), fall/drift while airborne (40% trigger), reduced idle (10%)
- Lighter `sanitize_grounded_idle_targets()` — only snap tiniest micro-bounces
- Re-recorded 500k: **32.2% idle+grounded, 28.1% airborne** (more fall data, less idle than v2)

**Training:**

- `tools/physics_sim.py` — Python port for closed-loop batches
- Teacher-forced 12-step rollout + closed-loop every 2nd batch
- Loss: tunnel penalty (25), airborne weight (4), idle weight (2), closed-loop weight (12)
- Output: `models/model_rollout_v3.bin` (256×256, 35 epochs)

**Status:** Interrupted (~2 epochs, resource limits). Still best playable: v2.

---

## Phase 10 — 3³ patch experiments (<1 hr train)

**Prompt:** Experiment with 3×3×3; training over 1 hour is unacceptable.

**Delivered:** `tools/patch3_experiments.py`, models in `models/patch3/`.

Input dim **39** (3³ + 12). 200k samples, center-crop from 9³ recordings. ~42 min total.

| Config | Train | C rollout pos ↓ | Vel err | Grounded mm% | Tunnel (N) | Step µs | × analytic |
|--------|-------|-----------------|---------|--------------|------------|---------|------------|
| *(analytic teacher)* | — | 0 | 0 | 0 | 0 | **~0.2** | **1×** |
| sweep baseline (128×128, 15 ep) | ~1.5m | 1.915 | 2.936 | 17.1 | 235 | 18 | ~90× |
| `128_rollout4_30ep` | 4m40s | 0.578 | 1.410 | 22.7 | 574 | 18 | ~90× |
| `256_rollout8_30ep` | 10m47s | 0.416 | 1.116 | 17.9 | 613 | 72 | ~360× |
| **`256_rollout8_idle4`** | **13m08s** | **0.335** | **0.890** | **9.9** | **231** | **72** | **~360×** |
| `64_rollout4_fast` | 10m13s | 0.476 | 1.336 | 12.7 | 215 | **5.7** | **~29×** |

Config notes:
- **256_rollout8_idle4** — 256×256, 8-step rollout, idle-weight 4 (v2-style) ← **fidelity anchor**
- **64_rollout4_fast** — 64×64, 4-step, 40 ep — speed floor (5.7 µs, 0.48 pos)
- Sweep 3³ underfit badly (1.91 pos); 30 ep + 256×256 + idle-weight fixes it (0.34)

**User verdict:** Fidelity satisfied — anchor for efficiency work.

**Play:** `./build/voxel_parkour.exe --model models/patch3/256_rollout8_idle4.bin`

---

## Phase 11 — Master model comparison vs analytic

**Prompt:** Compare every model against analytic baseline; lock efficiency target.

**Method:** `tools/bench_all_models.py` — 50 episodes × 300 steps, same inputs to analytic + neural. Analytic is oracle reference row (1×, ~0.2 µs/step). No runtime fallback.

### All models ranked by position error (pre Phase 12 C opts)

Method: 50 ep × 300 steps. Regenerate: `python tools/bench_all_models.py` (stdout only; does not overwrite this table).

| Model | Patch | MLP | Pos err | Vel err | Grounded mm% | Tunnel (N) | Step µs | × analytic |
|-------|-------|-----|---------|---------|--------------|------------|---------|------------|
| *(analytic teacher)* | — | — | 0 | 0 | 0 | 0 | **~0.2** | **1×** |
| `patch3/256_rollout8_idle4` ⭐ | 3³ | 256×256 | **0.335** | 0.890 | 9.9 | 231 | **72** | ~360× |
| `patch3/256_rollout8_30ep` | 3³ | 256×256 | 0.416 | 1.116 | 17.9 | 613 | 72 | ~360× |
| `patch3/64_rollout4_fast` | 3³ | 64×64 | 0.476 | 1.336 | 12.7 | 215 | 5.7 | ~29× |
| `model_rollout_v2` | 9³ | 256×256 | 0.530 | 1.060 | 13.2 | 868 | 268 | ~1340× |
| `patch_sweep/patch_2` | 2³ | 128×128 | 0.572 | 1.336 | 11.9 | 334 | 16 | ~80× |
| `model_rollout` | 9³ | 256×256 | 0.576 | 1.311 | 14.3 | 586 | 252 | ~1260× |
| `patch3/128_rollout4_30ep` | 3³ | 128×128 | 0.578 | 1.410 | 22.7 | 574 | 18 | ~90× |
| `model.bin` | 9³ | 128×128 | 0.632 | 1.647 | 14.8 | 798 | 145 | ~725× |
| `patch_sweep/patch_5` | 5³ | 128×128 | 0.809 | 1.729 | 11.9 | 170 | 29 | ~145× |
| `patch_sweep/patch_9` | 9³ | 128×128 | 0.847 | 2.123 | 18.1 | 194 | 118 | ~590× |
| `patch_sweep/patch_4` | 4³ | 128×128 | 0.863 | 2.142 | 17.5 | 222 | 22 | ~110× |
| `patch_sweep/patch_7` | 7³ | 128×128 | 1.028 | 2.462 | 32.2 | 384 | 61 | ~305× |
| `patch_sweep/patch_6` | 6³ | 128×128 | 1.523 | 2.314 | 13.7 | 160 | 42 | ~210× |
| `patch_sweep/patch_8` | 8³ | 128×128 | 1.882 | 3.292 | 32.2 | 212 | 85 | ~425× |
| `patch_sweep/patch_3` | 3³ | 128×128 | 1.915 | 2.936 | 17.1 | 235 | 18 | ~90× |

**Surprise:** 3³ 256×256 idle4 **beats 9³ v2 on C rollout** (0.34 vs 0.53) at ~3.7× lower latency.

**Efficiency frontier (next):**
1. Re-train **4³–5³** with 256×256 + idle4 recipe (sweep was under-trained)
2. **64×64** speed floor at 5.7 µs / 0.48 pos
3. Distill 9³ v2 → small patch for edge coverage
4. C inference: fused matmul, int8 weights

Post-opt full table: see Phase 12 below.

---

## Phase 12 — C inference optimizations

**Prompt:** Basic matmul/vector opts in C inference; verify fidelity and re-benchmark.

**Changes (`neural.c`, `CMakeLists.txt`):**
- 4-wide unrolled dot products with prefetch on next weight row
- Fused layer-1: `(x-mean)/std` combined with W1 matmul (eliminates norm scratch pass)
- Unrolled output denormalization
- Release flags: `-O3 -ffast-math -funroll-loops -march=native` (MinGW)

**Fidelity:** unchanged (same weights, same math order within float tolerance).

### All models post Phase 12 C opts (with inference speedup)

Method: 50 ep × 300 steps. Regenerate: `python tools/bench_all_models.py` (auto-updates this table).

<!-- bench-phase12-start -->
| Model | Patch | MLP | Pos err | Vel err | Grounded mm% | Tunnel (N) | Pre-opt µs | Post-opt µs | × analytic | Inf speedup |
|-------|-------|-----|---------|---------|--------------|------------|-------------|--------------|------------|-------------|
| *(analytic teacher)* | — | — | 0 | 0 | 0 | 0 | — | **~0.1** | **1×** | — | 
| `patch3/256_rollout8_idle4` ⭐ | 3³ | 256×256 | **0.335** | 0.890 | 9.9 | 231 | 72.0 | **11.9** | ~98× | **6.0×** |
| `patch3/256_rollout8_30ep` | 3³ | 256×256 | 0.416 | 1.116 | 17.9 | 613 | 72.0 | 12.0 | ~98× | 6.0× |
| `patch3/64_rollout4_fast` | 3³ | 64×64 | 0.476 | 1.336 | 12.7 | 215 | 5.7 | **1.9** | ~16× | **2.9×** |
| `model_rollout_v2` | 9³ | 256×256 | 0.530 | 1.060 | 13.2 | 868 | 268.0 | 54.9 | ~450× | 4.9× |
| `patch3/256_rollout12_25ep` | 3³ | 256×256 | 0.553 | 1.315 | 13.6 | 617 | — | 11.7 | ~96× | - |
| `patch_sweep/patch_2` | 2³ | 128×128 | 0.572 | 1.336 | 11.9 | 334 | 16.0 | 3.2 | ~26× | 5.0× |
| `model_rollout` | 9³ | 256×256 | 0.576 | 1.311 | 14.3 | 586 | 252.0 | 57.6 | ~472× | 4.4× |
| `patch3/128_rollout4_30ep` | 3³ | 128×128 | 0.578 | 1.410 | 22.7 | 574 | 18.0 | 4.2 | ~34× | 4.3× |
| `model` | 9³ | 128×128 | 0.632 | 1.647 | 14.8 | 798 | 145.0 | 25.4 | ~208× | 5.7× |
| `patch_sweep/patch_5` | 5³ | 128×128 | 0.810 | 1.734 | 12.0 | 170 | 29.0 | 7.2 | ~59× | 4.0× |
| `patch_sweep/patch_9` | 9³ | 128×128 | 0.847 | 2.123 | 18.1 | 194 | 118.0 | 25.9 | ~212× | 4.6× |
| `patch_sweep/patch_4` | 4³ | 128×128 | 0.863 | 2.142 | 17.5 | 222 | 22.0 | 6.2 | ~51× | 3.5× |
| `patch_sweep/patch_7` | 7³ | 128×128 | 1.028 | 2.462 | 32.2 | 384 | 61.0 | 13.9 | ~114× | 4.4× |
| `patch_sweep/patch_6` | 6³ | 128×128 | 1.523 | 2.315 | 13.7 | 160 | 42.0 | 10.0 | ~82× | 4.2× |
| `patch_sweep/patch_8` | 8³ | 128×128 | 1.881 | 3.276 | 32.1 | 209 | 85.0 | 19.2 | ~158× | 4.4× |
| `patch_sweep/patch_3` | 3³ | 128×128 | 1.915 | 2.936 | 17.1 | 235 | 18.0 | 4.2 | ~34× | 4.3× |
<!-- bench-phase12-end -->

### Key speedups (anchor models)

| Model | Pos err | Step µs | × analytic | Pre-opt | Inf speedup |
|-------|---------|---------|------------|---------|-------------|
| *(analytic teacher)* | 0 | **~0.1** | **1×** | — | — |
| `patch3/256_rollout8_idle4` ⭐ | 0.335 | **11.9** | **~98×** | 72 | **6.0×** |
| `patch3/64_rollout4_fast` | 0.476 | **1.9** | **~16×** | 5.7 | **2.9×** |
| `model_rollout_v2` (9³) | 0.530 | **54.9** | **~450×** | 268 | **4.9×** |

Anchor model now **11.9 µs/step** with best fidelity (0.335) — ~4.6× faster than post-opt 9³ v2 at better accuracy.

---

| #   | User prompt (summary)                                          |
| --- | -------------------------------------------------------------- |
| 1   | Implement spec, simple as possible                             |
| 2   | Plan neural swap: data → train → infer → benchmark             |
| 3   | Implement full pipeline until benchmarked                      |
| 4   | No fallbacks; fix spawn/goal; fail on missing model; retrain   |
| 5   | Goal still faulty — trigger on touch                           |
| 6   | Stability advice — reject A, pursue C + bigger model           |
| 7   | Implement rollout, logging, spec, progress log, start training |
| 8   | Document rollout; idle data + flat-rest targets for sink fix   |
| 9   | Patch size sweep 9³→2³ — benchmark inference vs accuracy       |
| 10  | Dataset v3 + closed-loop rollout training for phasing/fall fix |
| 11  | 3³ patch experiments — fast train configs under 1 hour         |
| 12  | Master model comparison; lock 3³ idle4 as efficiency anchor    |
| 13  | Consolidate benchmarks into progress-log (no separate md files) |
| 14  | C inference opts: unrolled matmul, fused norm, Release flags |
| 15  | Weight quantization sweep: int8/int4/fp16 on 5 diverse models |
| 16  | AVX2 SIMD final hyper-optimization (VNNI tested, FP32 AVX2 shipped) |


---


## Phase 13 — Weight quantization sweep

**Prompt:** Quantize 5 diverse models at several levels; compare speed and rollout fidelity.

**Schemes:** `fp32` (baseline), `int8_row` (per-row W8A32), `int8_layer` (per-tensor), `int4_row` (packed W4A32), `fp16` (half weights, FP32 compute).

Regenerate: `python tools/bench_quant.py`

<!-- bench-phase13-start -->
| Model | Quant | Size KB | Pos err | dpos vs fp32 | Vel err | Tunnel (N) | Step us | x analytic | vs fp32 speed |
|-------|-------|---------|---------|----------------|---------|------------|---------|------------|---------------|
| *(analytic teacher)* | — | — | 0 | — | 0 | 0 | **~0.1** | **1×** | — |
| 2³ sweep 128×128 | `fp32` | 79 | 0.572 | 0 | 1.336 | 334 | 4.6 | ~32× | 1.00× |
| 2³ sweep 128×128 | `int8_row` | 22 | 0.550 | -0.023 | 1.353 | 433 | 5.7 | ~39× | 0.80× |
| 2³ sweep 128×128 | `int8_layer` | 21 | 0.520 | -0.053 | 1.318 | 441 | 6.6 | ~45× | 0.70× |
| 2³ sweep 128×128 | `int4_row` | 12 | 1.439 | +0.866 | 2.857 | 964 | 50.4 | ~345× | 0.09× |
| 2³ sweep 128×128 | `fp16` | 40 | 0.591 | +0.019 | 1.397 | 339 | 4.5 | ~31× | 1.02× |
| 3³ anchor 256×256 | `fp32` | 304 | 0.335 | 0 | 0.890 | 231 | 12.1 | ~100× | 1.00× |
| 3³ anchor 256×256 | `int8_row` ** | 80 | 0.391 | +0.056 | 0.951 | 158 | 36.8 | ~302× | 0.33× |
| 3³ anchor 256×256 | `int8_layer` | 78 | 0.564 | +0.229 | 1.778 | 105 | 14.8 | ~121× | 0.82× |
| 3³ anchor 256×256 | `int4_row` | 42 | 0.658 | +0.323 | 2.119 | 511 | 339.5 | ~2783× | 0.04× |
| 3³ anchor 256×256 | `fp16` | 153 | 0.343 | +0.008 | 0.905 | 247 | 15.1 | ~123× | 0.81× |
| 3³ speed 64×64 | `fp32` | 28 | 0.476 | 0 | 1.336 | 215 | 2.8 | ~16× | 1.00× |
| 3³ speed 64×64 | `int8_row` | 8 | 0.479 | +0.002 | 1.281 | 199 | 2.7 | ~15× | 1.04× |
| 3³ speed 64×64 | `int8_layer` | 8 | 0.630 | +0.154 | 1.581 | 163 | 3.8 | ~21× | 0.75× |
| 3³ speed 64×64 | `int4_row` | 5 | 2.539 | +2.062 | 3.893 | 144 | 20.0 | ~111× | 0.14× |
| 3³ speed 64×64 | `fp16` | 15 | 0.464 | -0.013 | 1.334 | 223 | 2.7 | ~15× | 1.04× |
| 9³ full 256×256 | `fp32` | 1012 | 0.530 | 0 | 1.060 | 868 | 79.2 | ~377× | 1.00× |
| 9³ full 256×256 | `int8_row` | 261 | 0.582 | +0.052 | 1.362 | 1011 | 88.4 | ~421× | 0.90× |
| 9³ full 256×256 | `int8_layer` | 259 | 0.857 | +0.327 | 1.588 | 770 | 61.7 | ~294× | 1.28× |
| 9³ full 256×256 | `int4_row` | 136 | 4.765 | +4.235 | 3.338 | 72 | 919.8 | ~4380× | 0.09× |
| 9³ full 256×256 | `fp16` | 510 | 0.530 | 0 | 1.071 | 847 | 78.1 | ~372× | 1.01× |
| 9³ legacy 128×128 | `fp32` | 445 | 0.632 | 0 | 1.647 | 798 | 38.4 | ~211× | 1.00× |
| 9³ legacy 128×128 | `int8_row` | 117 | 0.672 | +0.040 | 1.512 | 768 | 41.6 | ~229× | 0.92× |
| 9³ legacy 128×128 | `int8_layer` | 116 | 1.033 | +0.401 | 1.721 | 1284 | 42.7 | ~234× | 0.90× |
| 9³ legacy 128×128 | `int4_row` | 63 | 13.454 | +12.822 | 9.668 | 299 | 396.0 | ~2176× | 0.10× |
| 9³ legacy 128×128 | `fp16` | 226 | 0.638 | +0.006 | 1.649 | 789 | 38.1 | ~209× | 1.01× |
<!-- bench-phase13-end -->

**Takeaways (5 models × 5 quant levels):**

| Scheme | Size | Fidelity (anchor) | Speed (CPU) | Verdict |
|--------|------|-------------------|-------------|---------|
| `fp16` | ~50% | +0.008 pos | ~same as fp32 | **Best tradeoff** — half file size, negligible drift |
| `int8_row` | ~26% | +0.056 pos | ~same (no SIMD VNNI yet) | Good for deployment size; needs int8 dot kernel for speed |
| `int8_layer` | ~26% | +0.229 pos | ~same | Too much drift for anchor; skip |
| `int4_row` | ~14% | +0.323 pos | **much slower** (nibble unpack) | Size win only; not viable without AVX512 + QAT |

**Models tested:** 3³ anchor (256×256), 3³ speed (64×64), 9³ v2 (256×256), 2³ sweep, 9³ legacy (128×128).

**Tools:** `tools/quantize_models.py` (export v2), `tools/bench_quant.py` (full sweep + table). Quantized files: `models/quant/{scheme}/`.

**Play quantized anchor:** `./build/voxel_parkour.exe --model models/quant/int8_row/patch3/256_rollout8_idle4.bin`

---


## Phase 14 — AVX2 SIMD inference (final hyper-optimization)

**Prompt:** Last push — AVX2-FMA FP32 matmul, VNNI int8 experiments, runtime CPU dispatch.

**Implementation:** `neural_simd.c` — AVX2 8-wide FP32 dots, fused layer-1 norm scratch, runtime `__builtin_cpu_supports` dispatch. VNNI W8A8 tested and rejected (rollout blow-up); int8 kept on scalar W8A32 path.

Regenerate: `python tools/bench_simd.py`

<!-- bench-phase14-start -->
| Config | Pos err | Forward us | Pre-SIMD us | Speedup | Kernel |
|--------|---------|------------|-------------|---------|--------|
| 3^3 anchor fp32 | 0.335 | 9.6 | 12.1 | 1.27x | avx2 |
| 3^3 anchor int8 | 0.391 | 14.4 | 14.0 | 0.97x | avx2 |
| 3^3 speed fp32 | 0.476 | 1.2 | 2.8 | 2.25x | avx2 |
| 3^3 speed int8 | 0.479 | 2.1 | 2.7 | 1.26x | avx2 |
| 9^3 v2 fp32 | 0.530 | 33.6 | 79.2 | 2.36x | avx2 |
| 9^3 v2 int8 | 0.582 | 54.7 | 88.4 | 1.62x | avx2 |
<!-- bench-phase14-end -->

**Takeaways:**
- **AVX2-FMA FP32** is the win: 9³ v2 **2.4× faster** (79→34 µs forward), 3³ speed **2.3×**, anchor **1.3×** — fidelity unchanged.
- **VNNI W8A8** (uint8 activations + int8 weights) caused rollout divergence — rejected. Real int8 speed needs QAT + integer dots, not post-hoc quant.
- **int8_row W8A32** keeps +0.056 pos drift on anchor; 9³ int8 speedup mostly from fixing scratch buffer sizing (741-dim norm was overflowing 256-float buf2).
- **End-to-end anchor:** 72 µs (Phase 11) → 12 µs (Phase 12 C opts) → **~10 µs** (Phase 14 AVX2) at **0.335 pos err** — **~7× faster** than pre-opt with best fidelity.

---

## Final statement

This project set out to replace analytic voxel physics with a pure neural `physics_step` — no fallbacks, no silent teacher at runtime — and prove the swap works through rollout benchmarks against the analytic oracle.

**What we proved:** A 3³-patch 256×256 MLP (`patch3/256_rollout8_idle4`) **beats 9³ full-context models on rollout fidelity** (0.335 vs 0.53 pos err) while running inference in **~10 µs/step** (~96× analytic). The bottleneck was never the idea; it was training recipe (rollout + idle weight), patch size, and inference implementation.

**Optimization arc (inference, anchor model):**

| Stage | Forward µs | vs analytic |
|-------|------------|-------------|
| Baseline FP32 (Phase 11) | 72 | ~360× |
| C unroll + fused norm (Phase 12) | 12 | ~60× |
| AVX2-FMA FP32 (Phase 14) | **~10** | **~50×** |
| int8_row quant (Phase 13) | ~14 | ~70× (4× smaller file) |

**What did not work:** int4 packing (slow scalar unpack), int8_layer (accuracy), VNNI without QAT (rollout blow-up). **fp16** is the best size/fidelity quant format if disk matters.

**Recommended production config:**
```bash
./build/voxel_parkour.exe --model models/patch3/256_rollout8_idle4.bin
```
For minimum size with negligible drift: `models/quant/fp16/patch3/256_rollout8_idle4.bin`.

**If continuing:** closed-loop v3 training for edge/fall behavior; 4³–5³ retrain with idle4 recipe; QAT int8 for genuine VNNI speed; GPU inference if sub-µs matters.

The engine is playable, benchmarked, documented, and pushed about as far as a scalar-then-SIMD C MLP on CPU reasonably goes without a new training pass.

---

## Hall of fame — superlatives (final final)

One-line awards from 16 phases, 15+ models, and a lot of wrong turns that turned out useful.

### Best overall
**`patch3/256_rollout8_idle4.bin`** — 0.335 rollout pos err, 9.9% grounded mismatch, ~10 µs/step (Phase 14). Beats every 9³ model on fidelity *and* speed. The config that won the project.

### Fastest (inference)
**`patch3/64_rollout4_fast.bin`** — **~1.2 µs** forward (Phase 14 AVX2), 0.476 pos err. A 64×64 MLP on a 39-dim 3³ patch. Still ~6× slower than analytic (~0.2 µs), but the neural floor.

### Best accuracy per microsecond
**3³ anchor fp32** — 0.335 pos / ~10 µs ≈ **0.034 err·µs⁻¹**. Nothing else comes close on the Pareto frontier we actually measured.

### Best training ROI
**`256_rollout8_idle4`** — 13 minutes train, 0.335 pos. Went from 1.915 (sweep 3³, 1.5 min) to 0.335 with one recipe change (256×256 + idle-weight 4 + 30 ep).

### Most improved (inference)
**Anchor model:** 72 µs → 12 µs (Phase 12) → **~10 µs** (Phase 14) = **~7×** total, fidelity unchanged. Phase 12 did the heavy lifting; AVX2 cleaned up the rest.

### Most improved (training)
**Rollout v0 → v2:** C rollout pos 0.68 → 0.42 (idle dataset). Sink fixed; new problems invented (falling, phasing).

### Best full-context model
**`model_rollout_v2.bin`** — 9³ 256×256, 0.530 pos err. The "see everything" baseline. Useful teacher; lost the efficiency war to a 3³ patch.

### Best quant format
**`fp16`** — half the file size, +0.008 pos drift on anchor, same speed as fp32. The adult in the room.

### Smallest useful model
**`quant/int8_row/patch3/64_rollout4_fast.bin`** — **~8 KB**, 0.479 pos err. Physics in the size of a GIF header.

### Smallest file (any)
**`quant/int4_row/patch3/64_rollout4_fast.bin`** — **~5 KB**. Also 2.5 pos err and unusable. Size isn't everything.

---

### Most unexpected
**3³ beats 9³ on rollout fidelity.** Less context, fewer inputs, better C benchmark score (0.335 vs 0.530). The whole "you need the full voxel patch" assumption died here.

### Weirdest result
**Patch sweep 2³ wins on pos err (0.392)** while 3³ gets 1.313 — same 128×128 architecture, different crop, opposite outcomes. Benchmark noise dressed as physics.

### Most cursed patch size
**8³** — 1.686 rollout pos in sweep; **7³** and **8³** in master comparison both >1.0. The patch size gremlin lives between 6³ and 9³.

### Best accidental result
**`int8_layer` on 2³ sweep** — 0.520 pos, *better* than fp32 (0.572). Per-tensor quant accidentally helping an under-trained model. Do not generalize.

### Biggest plot twist
Removing **52.7% analytic fallback** made rollout *worse* (0.23 → 0.68) — and that was the correct decision. Honest metrics over comfortable lies.

---

### Worst rollout (still shipped)
**`patch_sweep/patch_3.bin`** — 1.915 pos err. Same 3³ patch as the anchor, wrong train recipe. Proof the patch wasn't the problem.

### Worst quant disaster
**`int4_row` on 9³ legacy** — pos err **13.45**, 396 µs/step, file size "saved." Compressed the weights and decompressed the chaos.

### VNNI hall of shame
**W8A8 VNNI** — rollout pos err **inf** on first try. Fast integer dots that lie about physics. Rejected same day.

### Slowest "optimization"
**int4_row inference** — 339 µs forward on anchor (Phase 13). We made the model 4× smaller and **30× slower** with scalar nibble unpack.

### Most embarrassing bug
**741-dim norm into a 256-float `buf2`.** 9³ models silently corrupted until Phase 14. Classic off-by-485 buffer overflow. Fixed; trust restored.

### Never finished
**`model_rollout_v3.bin`** — closed-loop + edge-walk dataset, ~2 epochs before OOM. Still the best *idea* we didn't fully test. Playable winner remains v2/anchor.

---

### Most tunneling
**`model_rollout_v2`** — 868 tunnel steps / benchmark (neural, non-grounded overlap). Big model, big context, still phases.

### Least tunneling (good models)
**Anchor** — 231 tunnels. **64_rollout4_fast** — 215. Small patch + good train beats raw capacity.

### Highest grounded mismatch
**`patch_sweep/patch_7`** — 32.2%. **patch_8** — 32.2%. Patch sizes 7³ and 8³ agree on being confused about the floor.

---

### Fallback era (historical)
**Phase 3 `model.bin` with collision fallback** — 0.23 rollout pos, **52.7% of steps cheating** back to analytic. Best numbers, worst integrity. The villain origin story.

### Pure neural pain
**Phase 4 retrain** — 0.68 rollout pos, 941 tunnels, no fallback. The valley before rollout training.

### Rejected forever
**Option A: neural vel + analytic collision.** User said no. We meant it. Every temptation to "just snap to floor" or "just fix overlap in C" was rejected.

---

### Analytic baseline (the real speedrun)
**~0.1–0.2 µs/step**, 0 pos err, 0 tunnels, 1×. The teacher we replaced. Still **~50–100× faster** than our best neural. The game runs on a learned approximation of something that was never slow to begin with.

### The number that tells the story
**72 → 10 µs** on the anchor (7× faster), **0.335 pos err** (best we ever got), **13 min train**, **304 KB** (80 KB int8, 153 KB fp16). A tiny MLP pretending to be physics — and mostly succeeding.

---

## Expectation timeline

```
v0 engine only
  → neural replaces physics_step
    → benchmark proves swap works
      → NO fallbacks ever (hard requirement)
        → pure neural stability via training (rollout + scale)
          → compare models in benchmark (baseline vs rollout)
            → patch size sweep for inference speed tradeoffs
              → 3³ experiments: 256×256 idle4 beats 9³ v2 on fidelity
                → efficiency phase (4³–5³ retrain, distillation, inference opts)
                  → C matmul opts: 3³ anchor 72→12 µs (~98× analytic), fidelity unchanged
                    → quant sweep: fp16 best size/fidelity; int8_row for 4× smaller weights
                      → AVX2-FMA final push: anchor ~10 µs, 9³ v2 2.4× faster
```

---

## Files reference


| File                       | Role                                    |
| -------------------------- | --------------------------------------- |
| `docs/progress-log.md`     | **This file** — timeline, benchmarks, commands |
| `neural_simd.c`            | AVX2-FMA FP32 inference kernels + CPU dispatch |
| `docs/spec.md`             | living architecture spec                |
| `models/model.bin`         | 128×128 one-step baseline (pure neural) |
| `models/model_rollout.bin` | 256×256 rollout-trained |
| `models/model_rollout_v2.bin` | 9³ 256×256 — best full-context |
| `models/patch3/256_rollout8_idle4.bin` | **3³ fidelity anchor** — best C rollout (0.34) |
| `models/quant/{scheme}/` | Quantized v2 exports (int8_row, int8_layer, int4_row, fp16) |
| `models/patch3/best.bin` | Copy of fidelity anchor |
| `models/patch_sweep/patch_N.bin` | Patch sweep (128×128, 15 ep) |
| `data/train.bin`           | 500k samples (dataset v2/v3) |


---

## Commands cheat sheet

```bash
# Build
cd build && cmake .. -G "MinGW Makefiles" && cmake --build . --config Release

# Record
./build/voxel_parkour.exe --record 500000 --out data/train.bin

# Train rollout model (v2 — idle dataset)
python tools/train.py --data data/train.bin --out models/model_rollout_v2.bin \
  --hidden1 256 --hidden2 256 --rollout-steps 8 --rollout-weight 8.0 \
  --idle-weight 4.0 --epochs 30

# Train v3 — closed-loop + edge/fall dataset
python tools/train.py --data data/train.bin --out models/model_rollout_v3.bin \
  --hidden1 256 --hidden2 256 --rollout-steps 12 --rollout-weight 8.0 \
  --closed-loop-weight 12.0 --tunnel-penalty 25.0 --idle-weight 2.0 \
  --airborne-weight 4.0 --epochs 35 --batches-per-epoch 600 --batch 256 --cl-batch 48

# 3³ experiments (each <15 min)
python tools/patch3_experiments.py

# SIMD benchmark (Phase 14)
python tools/bench_simd.py

# Benchmark all models vs analytic
python tools/bench_all_models.py

# Quantize + benchmark 5 diverse models (int8/int4/fp16)
python tools/bench_quant.py

# Quantize only
python tools/quantize_models.py --all-defaults
python tools/quantize_models.py --in models/patch3/256_rollout8_idle4.bin --scheme int8_row

# Play fidelity anchor (3³)
./build/voxel_parkour.exe --model models/patch3/256_rollout8_idle4.bin

# Play full-context (9³)
./build/voxel_parkour.exe --model models/model_rollout_v2.bin
```

