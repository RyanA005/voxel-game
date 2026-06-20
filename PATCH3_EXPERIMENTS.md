# 3³ Patch Experiments

Input dim: 39 (3³ voxels + 12). 200k samples, center-crop from 9³ recordings.
Total wall time: **82m35s** (6 configs).

| Config | Train time | C rollout pos ↓ | C rollout vel | Grounded mm% | Tunnel | Neural µs |
|--------|------------|-----------------|---------------|--------------|--------|-----------|
| `128_rollout4_30ep` | 4m45s | 0.5776 | 1.410 | 22.7 | 574/12770 | 10.1 |
| `256_rollout8_30ep` | 10m51s | 0.4161 | 1.116 | 17.9 | 613/12534 | 41.5 |
| `256_rollout8_idle4` | 13m13s | 0.3351 | 0.890 | 9.9 | 231/12905 | 50.9 |
| `256_rollout12_25ep` | 32m09s | 0.5529 | 1.315 | 13.6 | 617/12613 | 11.8 |
| `512x256_rollout8` | 15m58s | 0.3066 | 0.873 | 13.5 | 686/13012 | 24.6 | **← best**
| `64_rollout4_fast` | 5m32s | 0.4249 | 1.134 | 10.5 | 191/12575 | 1.5 |

## Config details

- **128_rollout4_30ep** — 128×128, 4-step rollout, 30 ep (extended sweep baseline) (4m45s)
- **256_rollout8_30ep** — 256×256, 8-step rollout, 30 ep (10m51s)
- **256_rollout8_idle4** — 256×256, 8-step, idle-weight 4 (v2-style) (13m13s)
- **256_rollout12_25ep** — 256×256, 12-step rollout, 25 ep (v3-style teacher-forced only) (32m09s)
- **512x256_rollout8** — 512×256 wide, 8-step rollout, 25 ep (15m58s)
- **64_rollout4_fast** — 64×64 tiny, 4-step, 40 ep (speed baseline) (5m32s)

**Best C rollout:** `512x256_rollout8` (pos 0.3066)

Play: `./build/voxel_parkour.exe --model models/patch3/BEST.bin`

Sweep baseline (15 ep): `models/patch_sweep/patch_3.bin` — rollout pos 1.31
