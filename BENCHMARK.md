# Neural Physics Benchmark Results

Analytic baseline: **~0.08 µs** per step.

Dataset: native **3³** patch, 500k samples (`record_size=90`).

## Models kept (int8, 3³ patch)

| Model | Role | One-step grounded | Rollout grounded* | Neural step |
|-------|------|-------------------|-------------------|-------------|
| **model_q8_h24_p3.bin** ⭐ | **Default — best long-run stability** | 98.67% | **~89%** | ~1.3 µs |
| model_q8_h48x24_p3.bin | Larger two-layer | 98.74% | ~78% | ~3.5 µs |
| model_q8_h64_p3.bin | Largest single-layer | 98.65% | ~73% | ~2.8 µs |
| model_q8_h48_p3.bin | Wide single-layer | 98.69% | ~80% | ~2.4 µs |

\*Rollout = 50 episodes × 300 steps vs analytic physics (grounded agreement).

⭐ Game loads `model_q8_h24_p3.bin` first.

## Reproduce

```bash
cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build .
./voxel_parkour.exe --record 500000 --out ../data/train.bin

python tools/train.py --sweep              # speed ladder h12-h32
python tools/train.py --sweep-stable         # larger stable models (on 3³ data)

./voxel_parkour.exe --bench ../models/model_q8_h24_p3.bin
```

## Notes

- Bigger hidden layers and 5³ patches were tried; **h24 on 3³** still wins on rollout
- One-step val ≥98% for all models; multi-step drift remains without collision correction
- Old 7³/9³ models removed from `models/`
