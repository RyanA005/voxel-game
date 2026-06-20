#!/usr/bin/env python3
"""Fast 3³ patch experiments — each config targets <15 min train on CPU."""

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "tools" / "train.py"
EXE = ROOT / "build" / "voxel_parkour.exe"
DATA = ROOT / "data" / "train.bin"
OUT_DIR = ROOT / "models" / "patch3"

EXPERIMENTS = [
    {
        "name": "128_rollout4_30ep",
        "desc": "128×128, 4-step rollout, 30 ep (extended sweep baseline)",
        "args": ["--hidden1", "128", "--hidden2", "128", "--rollout-steps", "4",
                 "--rollout-weight", "4.0", "--idle-weight", "2.0", "--epochs", "30",
                 "--batches-per-epoch", "400"],
    },
    {
        "name": "256_rollout8_30ep",
        "desc": "256×256, 8-step rollout, 30 ep",
        "args": ["--hidden1", "256", "--hidden2", "256", "--rollout-steps", "8",
                 "--rollout-weight", "8.0", "--idle-weight", "2.0", "--epochs", "30",
                 "--batches-per-epoch", "400"],
    },
    {
        "name": "256_rollout8_idle4",
        "desc": "256×256, 8-step, idle-weight 4 (v2-style)",
        "args": ["--hidden1", "256", "--hidden2", "256", "--rollout-steps", "8",
                 "--rollout-weight", "8.0", "--idle-weight", "4.0", "--epochs", "30",
                 "--batches-per-epoch", "400"],
    },
    {
        "name": "256_rollout12_25ep",
        "desc": "256×256, 12-step rollout, 25 ep (v3-style teacher-forced only)",
        "args": ["--hidden1", "256", "--hidden2", "256", "--rollout-steps", "12",
                 "--rollout-weight", "8.0", "--idle-weight", "2.0", "--airborne-weight", "4.0",
                 "--epochs", "25", "--batches-per-epoch", "500", "--no-closed-loop"],
    },
    {
        "name": "512x256_rollout8",
        "desc": "512×256 wide, 8-step rollout, 25 ep",
        "args": ["--hidden1", "512", "--hidden2", "256", "--rollout-steps", "8",
                 "--rollout-weight", "8.0", "--idle-weight", "2.0", "--epochs", "25",
                 "--batches-per-epoch", "400"],
    },
    {
        "name": "64_rollout4_fast",
        "desc": "64×64 tiny, 4-step, 40 ep (speed baseline)",
        "args": ["--hidden1", "64", "--hidden2", "64", "--rollout-steps", "4",
                 "--rollout-weight", "4.0", "--idle-weight", "2.0", "--epochs", "40",
                 "--batches-per-epoch", "300"],
    },
]


def run_train(name: str, extra_args: list) -> tuple[Path, float]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{name}.bin"
    cmd = [
        sys.executable, "-u", str(TRAIN),
        "--data", str(DATA),
        "--out", str(out),
        "--patch", "3",
        "--batch", "256",
        "--max-samples", "200000",
        *extra_args,
    ]
    print(f"\n=== {name} ===", flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True, cwd=ROOT)
    return out, time.time() - t0


def run_bench(model_path: Path) -> dict:
    text = subprocess.check_output([str(EXE), "--bench", str(model_path)], cwd=ROOT, text=True)

    def grab(pattern, default=0.0):
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
        "tunnel_steps": int(tunnel.group(2)) if tunnel else 0,
    }


def fmt_time(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def main():
    if not DATA.exists() or not EXE.exists():
        print("Need data/train.bin and build/voxel_parkour.exe", file=sys.stderr)
        sys.exit(1)

    rows = []
    total_t0 = time.time()

    for exp in EXPERIMENTS:
        model, train_sec = run_train(exp["name"], exp["args"])
        stats = run_bench(model)
        rows.append({"name": exp["name"], "desc": exp["desc"], "train_sec": train_sec, **stats})

    total_sec = time.time() - total_t0
    best = min(rows, key=lambda r: r["rollout_pos"])

    print(f"\n=== 3³ experiments ({fmt_time(total_sec)} total) ===")
    print("| Config | Train | Rollout pos | Neural µs |")
    print("|--------|-------|-------------|-----------|")
    for r in rows:
        mark = " ← best" if r["name"] == best["name"] else ""
        print(
            f"| {r['name']} | {fmt_time(r['train_sec'])} | {r['rollout_pos']:.4f} | "
            f"{r['neural_us']:.1f} |{mark}"
        )
    print(f"\nBest: {best['name']} ({best['rollout_pos']:.4f})")
    print("Copy results into progress-log.md Phase 10.")


if __name__ == "__main__":
    main()
