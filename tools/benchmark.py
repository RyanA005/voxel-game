#!/usr/bin/env python3
"""Offline one-step and short rollout benchmark."""

import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from train import INPUT_DIM, OUTPUT_DIM, HIDDEN1, HIDDEN2, MLP, load_dataset, split_by_seed, denorm  # noqa: E402


def load_model_bin(path: Path):
    data = path.read_bytes()
    magic, version = struct.unpack_from("<II", data, 0)
    assert magic == 0x214D4C50 and version == 1
    dims = struct.unpack_from("<4i", data, 8)
    off = 24
    in_dim, h1, h2, out_dim = dims

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
    w2 = readf(h2 * h1).reshape(h2, h1)
    b2 = readf(h2)
    w3 = readf(out_dim * h2).reshape(out_dim, h2)
    b3 = readf(out_dim)

    model = MLP()
    with torch.no_grad():
        model.net[0].weight.copy_(torch.from_numpy(w1))
        model.net[0].bias.copy_(torch.from_numpy(b1))
        model.net[2].weight.copy_(torch.from_numpy(w2))
        model.net[2].bias.copy_(torch.from_numpy(b2))
        model.net[4].weight.copy_(torch.from_numpy(w3))
        model.net[4].bias.copy_(torch.from_numpy(b3))
    return model, x_mean, x_std, y_mean, y_std


def main():
    data_path = ROOT / "data" / "train.bin"
    model_path = ROOT / "models" / "model.bin"
    xs, ys, seeds = load_dataset(data_path)
    model, x_mean, x_std, y_mean, y_std = load_model_bin(model_path)

    val_mask = split_by_seed(seeds)
    x_val, y_val = xs[val_mask], ys[val_mask]
    x_n = ((x_val - x_mean) / x_std).astype(np.float32)

    with torch.no_grad():
        pred_n = model(torch.from_numpy(x_n)).numpy()
    pred = denorm(pred_n, y_mean, y_std)

    pos_rmse = np.sqrt(((pred[:, :3] - y_val[:, :3]) ** 2).mean())
    vel_rmse = np.sqrt(((pred[:, 3:6] - y_val[:, 3:6]) ** 2).mean())
    g_acc = ((pred[:, 6] >= 0.5) == (y_val[:, 6] >= 0.5)).mean()
    pos_p95 = np.percentile(np.linalg.norm(pred[:, :3] - y_val[:, :3], axis=1), 95)

    print("=== Python One-Step Benchmark (val set) ===")
    print(f"Samples:           {len(x_val)}")
    print(f"Position RMSE:     {pos_rmse:.6f}")
    print(f"Position p95:      {pos_p95:.6f}")
    print(f"Velocity RMSE:     {vel_rmse:.6f}")
    print(f"Grounded accuracy: {g_acc:.4f}")

    exe = ROOT / "build" / "voxel_parkour.exe"
    if exe.exists():
        print("\n=== C Rollout Benchmark ===")
        subprocess.run([str(exe), "--bench", str(model_path)], check=False)


if __name__ == "__main__":
    main()
