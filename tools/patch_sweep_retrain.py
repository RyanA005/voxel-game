#!/usr/bin/env python3
"""Retrain patch sweep (9³→2³) with the 256_rollout8_idle4 recipe for comparison."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "tools" / "train.py"
EXE = ROOT / "build" / "voxel_parkour.exe"
DATA = ROOT / "data" / "train.bin"
OUT_DIR = ROOT / "models" / "patch_sweep_retrain"

# Old sweep (patch_sweep.py) for side-by-side reference — from progress-log Phase 8 re-benchmark.
SWEEP_BASELINE = {
    9: {"rollout_pos": 0.6753, "rollout_vel": 2.083, "grounded_mm": 14.7, "tunnel": 437, "neural_us": 67.4},
    8: {"rollout_pos": 1.6862, "rollout_vel": 3.510, "grounded_mm": 26.8, "tunnel": 521, "neural_us": 50.1},
    7: {"rollout_pos": 0.6903, "rollout_vel": 2.128, "grounded_mm": 21.8, "tunnel": 314, "neural_us": 38.6},
    6: {"rollout_pos": 1.2078, "rollout_vel": 2.064, "grounded_mm": 10.8, "tunnel": 128, "neural_us": 22.5},
    5: {"rollout_pos": 0.6270, "rollout_vel": 1.661, "grounded_mm": 9.5, "tunnel": 276, "neural_us": 15.1},
    4: {"rollout_pos": 0.5384, "rollout_vel": 1.617, "grounded_mm": 11.9, "tunnel": 111, "neural_us": 12.5},
    3: {"rollout_pos": 1.3130, "rollout_vel": 2.476, "grounded_mm": 12.2, "tunnel": 283, "neural_us": 9.7},
    2: {"rollout_pos": 0.3920, "rollout_vel": 1.403, "grounded_mm": 9.1, "tunnel": 316, "neural_us": 8.8},
}

PROFILES = {
    "idle4": {
        "desc": "256_rollout8_idle4 (exact)",
        "epochs": 30,
        "batches_per_epoch": 400,
        "max_samples": 200_000,
    },
    "fast": {
        "desc": "idle4 recipe, reduced budget (~25 min total on CPU)",
        "epochs": 20,
        "batches_per_epoch": 250,
        "max_samples": 120_000,
    },
}


def patch_input_dim(n: int) -> int:
    return n**3 + 12


def run_train(patch_n: int, profile: dict) -> tuple[Path, float]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"patch_{patch_n}.bin"
    cmd = [
        sys.executable,
        "-u",
        str(TRAIN),
        "--data",
        str(DATA),
        "--out",
        str(out),
        "--patch",
        str(patch_n),
        "--hidden1",
        "256",
        "--hidden2",
        "256",
        "--rollout-steps",
        "8",
        "--rollout-weight",
        "8.0",
        "--idle-weight",
        "4.0",
        "--epochs",
        str(profile["epochs"]),
        "--batches-per-epoch",
        str(profile["batches_per_epoch"]),
        "--batch",
        "256",
        "--max-samples",
        str(profile["max_samples"]),
    ]
    print(f"\n=== patch {patch_n}³ idle4 retrain ===", flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True, cwd=ROOT)
    return out, time.time() - t0


def run_bench(model_path: Path) -> dict:
    text = subprocess.check_output([str(EXE), "--bench", str(model_path)], cwd=ROOT, text=True)

    def grab(pattern: str, default: float = 0.0) -> float:
        m = re.search(pattern, text)
        return float(m.group(1)) if m else default

    tunnel = re.search(r"Tunnel \(neural\):\s+(\d+) / (\d+)", text)
    return {
        "rollout_pos": grab(r"Mean position error:\s+([\d.]+)"),
        "rollout_vel": grab(r"Mean velocity error:\s+([\d.]+)"),
        "grounded_mm": grab(r"Grounded mismatch:\s+([\d.]+)"),
        "neural_us": grab(r"Avg neural step:\s+([\d.]+)"),
        "forward_us": grab(r"Forward-only \(10k avg\):\s+([\d.]+)"),
        "tunnel": int(tunnel.group(1)) if tunnel else 0,
    }


def fmt_time(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--profile",
        choices=list(PROFILES),
        default="fast",
        help="idle4=full recipe; fast=shorter run for storyline comparison",
    )
    ap.add_argument(
        "--patches",
        type=str,
        default="9-2",
        help="e.g. 9-2 (default), 3, or 4,5,6",
    )
    args = ap.parse_args()
    profile = PROFILES[args.profile]

    if not DATA.exists() or not EXE.exists():
        print("Need data/train.bin and build/voxel_parkour.exe", file=sys.stderr)
        sys.exit(1)

    if args.patches == "9-2":
        patches = list(range(9, 1, -1))
    else:
        patches = [int(x.strip()) for x in args.patches.split(",")]

    rows = []
    total_t0 = time.time()

    for n in patches:
        model, train_sec = run_train(n, profile)
        stats = run_bench(model)
        old = SWEEP_BASELINE[n]
        rows.append(
            {
                "patch": n,
                "input_dim": patch_input_dim(n),
                "train_sec": train_sec,
                "old_pos": old["rollout_pos"],
                "delta_pos": old["rollout_pos"] - stats["rollout_pos"],
                **stats,
            }
        )

    total_sec = time.time() - total_t0
    print(f"\n=== Patch retrain ({profile['desc']}) — {fmt_time(total_sec)} ===")
    print(
        f"Train budget: {profile['epochs']} ep × {profile['batches_per_epoch']} batches, "
        f"{profile['max_samples']:,} samples, 256×256, rollout 8, idle-weight 4"
    )
    print()
    print(
        "| Patch | Input | Train | sweep pos | idle4 pos | Δ pos | vel | gnd mm% | tunnel | step µs |"
    )
    print(
        "|-------|-------|-------|-----------|-----------|-------|-----|---------|--------|---------|"
    )
    for r in rows:
        print(
            f"| {r['patch']}³ | {r['input_dim']} | {fmt_time(r['train_sec'])} | "
            f"{r['old_pos']:.4f} | {r['rollout_pos']:.4f} | {r['delta_pos']:+.4f} | "
            f"{r['rollout_vel']:.3f} | {r['grounded_mm']:.1f} | {r['tunnel']} | {r['neural_us']:.1f} |"
        )

    best_new = min(rows, key=lambda r: r["rollout_pos"])
    worst_old = max(rows, key=lambda r: r["old_pos"])
    print(
        f"\nBest retrained: {best_new['patch']}³ @ {best_new['rollout_pos']:.4f} pos "
        f"(sweep was {best_new['old_pos']:.4f})"
    )
    print(
        f"Largest sweep outlier fixed: {worst_old['patch']}³ "
        f"{worst_old['old_pos']:.4f} → {worst_old['rollout_pos']:.4f} "
        f"(Δ {worst_old['delta_pos']:+.4f})"
    )


if __name__ == "__main__":
    main()
