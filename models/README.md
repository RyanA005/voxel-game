# Models

Binary weight files for the C MLP runtime (`neural.c`). Not tracked in git — train or download separately.

## Recommended defaults

| Use case | Path | Patch | Hidden | Rollout pos | Step time |
|----------|------|-------|--------|-------------|-----------|
| **Play (default)** | `patch3/256_rollout8_idle4.bin` | 3³ | 256×256 | 0.335 | ~10 µs |
| Speed floor | `patch3/64_rollout4_fast.bin` | 3³ | 64×64 | 0.425 | ~1 µs |
| Best accuracy (3³) | `patch3/512x256_rollout8.bin` | 3³ | 512×256 | 0.307 | ~17 µs |
| Full context (9³) | `model_rollout_v2.bin` | 9³ | 256×256 | 0.530 | ~31 µs |

```bash
./build/voxel_parkour --model models/patch3/256_rollout8_idle4.bin
./build/voxel_parkour --bench models/patch3/256_rollout8_idle4.bin
```

## Directory layout

```
models/
├── model.bin                  # 9³ legacy baseline (128×128, one-step trained)
├── model_rollout.bin          # 9³ 256×256 rollout v1
├── model_rollout_v2.bin       # 9³ 256×256 rollout v2 (best full-context)
├── patch3/                    # 3³ experiment sweep (256×256 family)
├── patch_sweep/               # patch size 2³–9³ at 128×128 (under-trained baseline)
├── patch_sweep_retrain/       # patch 3³–9³ retrained with idle4 recipe (256×256)
└── quant/                     # quantized v2 exports (int8/int4/fp16)
    ├── int8_row/
    ├── int8_layer/
    ├── int4_row/
    └── fp16/
```

## Training new models

Requires `data/train.bin` (record with `--record`).

```bash
./build/voxel_parkour --record 500000 --out data/train.bin

python tools/train.py --data data/train.bin \
  --out models/patch3/256_rollout8_idle4.bin \
  --patch 3 --hidden1 256 --hidden2 256 \
  --rollout-steps 8 --epochs 30 --idle-weight 4
```

See `tools/train.py --help` for rollout weight, sample weighting, and export options.

## Quantization

```bash
python tools/quantize_models.py --all-defaults
python tools/bench_quant.py
```

Quantized models mirror the FP32 tree under `models/quant/<scheme>/`.

## Benchmarks

```bash
python tools/bench_all_models.py    # all .bin files vs analytic teacher
python tools/bench_simd.py            # AVX2 kernel comparison
```

Results are written to `docs/progress-log.md`.
