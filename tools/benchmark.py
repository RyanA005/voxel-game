#!/usr/bin/env python3
"""Offline one-step benchmark + optional C rollout."""

import argparse
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from train import INPUT_DIM, OUTPUT_DIM, MLP, load_dataset, split_by_seed, fmt_duration, log  # noqa: E402


def denorm(pred_n, y_mean, y_std):
    return pred_n * y_std + y_mean


def load_float_model(path: Path):
    data = path.read_bytes()
    magic, version = struct.unpack_from("<II", data, 0)
    assert magic == 0x214D4C50 and version == 1, f"expected float v1 model, got version {version}"
    in_dim, h1, h2, out_dim = struct.unpack_from("<4i", data, 8)
    assert in_dim == INPUT_DIM and out_dim == OUTPUT_DIM

    off = 24

    def readf(n):
        nonlocal off
        arr = np.frombuffer(data, dtype=np.float32, count=n, offset=off)
        off += n * 4
        return arr.copy()

    x_mean = readf(in_dim)
    x_std = readf(in_dim)
    y_mean = readf(out_dim)
    y_std = readf(out_dim)
    w1 = readf(h1 * in_dim).reshape(h1, in_dim)
    b1 = readf(h1)
    w2 = readf(h2 * h1).reshape(h2, h1) if h2 > 0 else None
    b2 = readf(h2) if h2 > 0 else None
    h_out = h2 if h2 > 0 else h1
    w3 = readf(out_dim * h_out).reshape(out_dim, h_out)
    b3 = readf(out_dim)

    model = MLP(h1, h2)
    with torch.no_grad():
        model.net[0].weight.copy_(torch.from_numpy(w1))
        model.net[0].bias.copy_(torch.from_numpy(b1))
        if h2 > 0:
            model.net[2].weight.copy_(torch.from_numpy(w2))
            model.net[2].bias.copy_(torch.from_numpy(b2))
            model.net[4].weight.copy_(torch.from_numpy(w3))
            model.net[4].bias.copy_(torch.from_numpy(b3))
        else:
            model.net[2].weight.copy_(torch.from_numpy(w3))
            model.net[2].bias.copy_(torch.from_numpy(b3))
    return model, x_mean, x_std, y_mean, y_std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "train.bin"))
    ap.add_argument("--model", default=str(ROOT / "models" / "model.bin"))
    ap.add_argument("--no-c", action="store_true", help="skip C rollout benchmark")
    args = ap.parse_args()

    t0 = time.perf_counter()
    data_path = Path(args.data)
    model_path = Path(args.model)

    log(f"Loading dataset {data_path} ...")
    xs, ys, seeds = load_dataset(data_path)

    log(f"Loading model {model_path.name} ...")
    model, x_mean, x_std, y_mean, y_std = load_float_model(model_path)

    val_mask = split_by_seed(seeds)
    x_val, y_val = xs[val_mask], ys[val_mask]
    x_n = ((x_val - x_mean) / x_std).astype(np.float32)

    log(f"Running one-step val forward on {len(x_val):,} samples ...")
    with torch.no_grad():
        pred_n = model(torch.from_numpy(x_n)).numpy()
    pred = denorm(pred_n, y_mean, y_std)

    pos_rmse = np.sqrt(((pred[:, :3] - y_val[:, :3]) ** 2).mean())
    vel_rmse = np.sqrt(((pred[:, 3:6] - y_val[:, 3:6]) ** 2).mean())
    g_acc = ((pred[:, 6] >= 0.5) == (y_val[:, 6] >= 0.5)).mean()
    pos_p95 = np.percentile(np.linalg.norm(pred[:, :3] - y_val[:, :3], axis=1), 95)

    log(f"=== Python one-step benchmark ({fmt_duration(time.perf_counter() - t0)}) ===")
    log(f"Samples:           {len(x_val):,}")
    log(f"Position RMSE:     {pos_rmse:.6f}")
    log(f"Position p95:      {pos_p95:.6f}")
    log(f"Velocity RMSE:     {vel_rmse:.6f}")
    log(f"Grounded accuracy: {g_acc:.4f}")

    exe = ROOT / "build" / "voxel_parkour.exe"
    if not args.no_c and exe.exists() and model_path.suffix == ".bin":
        log("\n=== C rollout benchmark ===")
        subprocess.run([str(exe), "--bench", str(model_path)], check=False)
    elif model_path.suffix == ".bin" and "q8" in model_path.name:
        log("(C rollout skipped: int8 models — use build/voxel_parkour.exe --bench directly)")


if __name__ == "__main__":
    main()
