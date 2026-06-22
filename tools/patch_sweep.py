#!/usr/bin/env python3
"""Train and benchmark MLP models for patch sizes 9³ down to 2³."""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "tools" / "train.py"
EXE = ROOT / "build" / "voxel_parkour.exe"
DATA = ROOT / "data" / "train.bin"
OUT_DIR = ROOT / "models" / "patch_sweep"


def patch_input_dim(n: int) -> int:
    return n ** 3 + 12


def run_train(patch_n: int, epochs: int = 15) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"patch_{patch_n}.bin"
    cmd = [
        sys.executable,
        str(TRAIN),
        "--data", str(DATA),
        "--out", str(out),
        "--patch", str(patch_n),
        "--hidden1", "128",
        "--hidden2", "128",
        "--rollout-steps", "4",
        "--rollout-weight", "4.0",
        "--idle-weight", "2.0",
        "--epochs", str(epochs),
        "--batches-per-epoch", "400",
        "--batch", "256",
        "--max-samples", "200000",
    ]
    print(f"\n=== Training patch {patch_n}³ (input {patch_input_dim(patch_n)}) ===", flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)
    return out


def run_bench(model_path: Path) -> dict:
    cmd = [str(EXE), "--bench", str(model_path)]
    text = subprocess.check_output(cmd, cwd=ROOT, text=True)
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


def main():
    if not DATA.exists():
        print(f"Missing {DATA} — run recorder first", file=sys.stderr)
        sys.exit(1)
    if not EXE.exists():
        print(f"Missing {EXE} — build first", file=sys.stderr)
        sys.exit(1)

    patches = list(range(9, 1, -1))  # 9,8,...,2
    rows = []

    for n in patches:
        model = run_train(n)
        stats = run_bench(model)
        params = 128 * patch_input_dim(n) + 128 * 128 + 128 * 7  # rough
        rows.append({
            "patch": n,
            "input_dim": patch_input_dim(n),
            "params_approx": params,
            **stats,
        })

    print("\n=== Patch sweep summary ===")
    print("| Patch | Input dim | Rollout pos | Neural µs |")
    print("|-------|-----------|-------------|-----------|")
    for r in rows:
        print(
            f"| {r['patch']}³ | {r['input_dim']} | {r['rollout_pos']:.4f} | {r['neural_us']:.1f} |"
        )
    print("\nCopy results into docs/progress-log.md Phase 8.")


if __name__ == "__main__":
    main()
