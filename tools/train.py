#!/usr/bin/env python3
"""Train, quantize, and export physics MLP models."""

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SCALARS = 12
OUTPUT_DIM = 7
TAIL_BYTES = 63  # bytes after voxels[] in TrainingRecord
MODEL_MAGIC = 0x214D4C50
MODEL_VER_FLOAT = 1
MODEL_VER_INT8 = 2
DATASET_MAGIC = 0x4B435056


def make_record_dtype(voxel_count: int):
    dt = np.dtype(
        [
            ("voxels", np.uint8, voxel_count),
            ("offset", np.float32, 3),
            ("vel", np.float32, 3),
            ("grounded", np.uint8),
            ("input", np.uint8, 5),
            ("dt", np.float32),
            ("target_delta", np.float32, 3),
            ("target_vel", np.float32, 3),
            ("target_grounded", np.uint8),
            ("seed", np.uint32),
        ]
    )
    assert dt.itemsize == voxel_count + TAIL_BYTES
    return dt


def fmt_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def log(msg: str) -> None:
    print(msg, flush=True)


def load_dataset(path: Path):
    t0 = time.perf_counter()
    raw = Path(path).read_bytes()
    magic, version, record_size = struct.unpack_from("<IHH", raw, 0)
    assert magic == DATASET_MAGIC, f"bad magic {magic:#x}"
    voxel_count = record_size - TAIL_BYTES
    assert voxel_count > 0 and voxel_count + TAIL_BYTES == record_size

    n = (len(raw) - 8) // record_size
    rec_dtype = make_record_dtype(voxel_count)
    recs = np.frombuffer(raw, dtype=rec_dtype, offset=8, count=n)

    voxels = recs["voxels"].astype(np.float32) / 4.0
    vel_n = np.column_stack(
        (recs["vel"][:, 0] / 6.0, recs["vel"][:, 1] / 10.0, recs["vel"][:, 2] / 6.0)
    ).astype(np.float32)
    xs = np.concatenate(
        [
            voxels,
            recs["offset"].astype(np.float32),
            vel_n,
            recs["grounded"].astype(np.float32)[:, None],
            recs["input"].astype(np.float32),
        ],
        axis=1,
    )
    ys = np.concatenate(
        [
            recs["target_delta"].astype(np.float32),
            recs["target_vel"].astype(np.float32),
            recs["target_grounded"].astype(np.float32)[:, None],
        ],
        axis=1,
    )
    seeds = recs["seed"].astype(np.uint32)

    elapsed = time.perf_counter() - t0
    patch = round(voxel_count ** (1.0 / 3.0))
    log(
        f"Loaded {n:,} samples from {path} ({len(raw) / 1e6:.1f} MB, "
        f"{patch}^3 patch, input={xs.shape[1]}) in {elapsed:.2f}s"
    )
    return xs, ys, seeds


class MLP(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2=0):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden1), nn.ReLU()]
        if hidden2 > 0:
            layers += [nn.Linear(hidden1, hidden2), nn.ReLU(), nn.Linear(hidden2, OUTPUT_DIM)]
        else:
            layers += [nn.Linear(hidden1, OUTPUT_DIM)]
        self.net = nn.Sequential(*layers)
        self.hidden2 = hidden2

    def forward(self, x):
        return self.net(x)


def split_by_seed(seeds, val_ratio=0.2):
    unique = np.unique(seeds)
    rng = np.random.default_rng(42)
    rng.shuffle(unique)
    n_val = max(1, int(len(unique) * val_ratio))
    val_seeds = set(unique[:n_val].tolist())
    return np.array([s in val_seeds for s in seeds])


def get_weights(model, hidden2):
    state = model.state_dict()
    w1 = state["net.0.weight"].cpu().numpy().astype(np.float32)
    b1 = state["net.0.bias"].cpu().numpy().astype(np.float32)
    if hidden2 > 0:
        w2 = state["net.2.weight"].cpu().numpy().astype(np.float32)
        b2 = state["net.2.bias"].cpu().numpy().astype(np.float32)
        w3 = state["net.4.weight"].cpu().numpy().astype(np.float32)
        b3 = state["net.4.bias"].cpu().numpy().astype(np.float32)
    else:
        w2 = b2 = None
        w3 = state["net.2.weight"].cpu().numpy().astype(np.float32)
        b3 = state["net.2.bias"].cpu().numpy().astype(np.float32)
    return w1, b1, w2, b2, w3, b3


def fuse_weights(w1, b1, w2, b2, w3, b3, x_mean, x_std, y_mean, y_std):
    w1 = w1 / x_std
    b1 = b1 - w1 @ x_mean
    w3 = w3 * y_std[:, None]
    b3 = b3 * y_std + y_mean
    return w1, b1, w2, b2, w3, b3


def quantize_rows(w: np.ndarray):
    scales = np.max(np.abs(w), axis=1) / 127.0
    scales[scales < 1e-8] = 1.0
    q = np.clip(np.round(w / scales[:, None]), -127, 127).astype(np.int8)
    return q, scales.astype(np.float32)


def export_float(path, w1, b1, w2, b2, w3, b3, x_mean, x_std, y_mean, y_std, input_dim, h1, h2):
    path.parent.mkdir(parents=True, exist_ok=True)
    w2 = w2 if w2 is not None else np.zeros((0,), dtype=np.float32)
    b2 = b2 if b2 is not None else np.zeros((0,), dtype=np.float32)
    with path.open("wb") as f:
        f.write(struct.pack("<II", MODEL_MAGIC, MODEL_VER_FLOAT))
        f.write(struct.pack("<4i", input_dim, h1, h2, OUTPUT_DIM))
        f.write(x_mean.astype(np.float32).tobytes())
        f.write(x_std.astype(np.float32).tobytes())
        f.write(y_mean.astype(np.float32).tobytes())
        f.write(y_std.astype(np.float32).tobytes())
        f.write(w1.tobytes())
        f.write(b1.tobytes())
        f.write(w2.tobytes())
        f.write(b2.tobytes())
        f.write(w3.tobytes())
        f.write(b3.tobytes())


def export_int8(path, w1, b1, w2, b2, w3, b3, input_dim, h1, h2):
    path.parent.mkdir(parents=True, exist_ok=True)
    q1, s1 = quantize_rows(w1)
    q3, s3 = quantize_rows(w3)
    with path.open("wb") as f:
        f.write(struct.pack("<II", MODEL_MAGIC, MODEL_VER_INT8))
        f.write(struct.pack("<4i", input_dim, h1, h2, OUTPUT_DIM))
        f.write(q1.tobytes())
        f.write(s1.tobytes())
        f.write(b1.tobytes())
        if h2 > 0:
            q2, s2 = quantize_rows(w2)
            f.write(q2.tobytes())
            f.write(s2.tobytes())
            f.write(b2.tobytes())
        f.write(q3.tobytes())
        f.write(s3.tobytes())
        f.write(b3.tobytes())


def metrics(pred_n, y_n, y_mean, y_std):
    pred = pred_n * y_std + y_mean
    target = y_n * y_std + y_mean
    pos = float(np.sqrt(((pred[:, :3] - target[:, :3]) ** 2).mean()))
    vel = float(np.sqrt(((pred[:, 3:6] - target[:, 3:6]) ** 2).mean()))
    g = float(((pred[:, 6] >= 0.5) == (target[:, 6] >= 0.5)).mean())
    return pos, vel, g


def train_one(args, hidden1, hidden2, out_path, int8=False, dataset=None):
    if dataset is None:
        xs, ys, seeds = load_dataset(Path(args.data))
    else:
        xs, ys, seeds = dataset

    val_mask = split_by_seed(seeds)
    x_train, y_train = xs[~val_mask], ys[~val_mask]
    x_val, y_val = xs[val_mask], ys[val_mask]

    x_mean = x_train.mean(axis=0, dtype=np.float32)
    x_std = x_train.std(axis=0, dtype=np.float32)
    x_std[x_std < 1e-6] = 1.0
    y_mean = y_train.mean(axis=0, dtype=np.float32)
    y_std = y_train.std(axis=0, dtype=np.float32)
    y_std[y_std < 1e-6] = 1.0

    x_train_n = (x_train - x_mean) / x_std
    y_train_n = (y_train - y_mean) / y_std
    x_val_n = (x_val - x_mean) / x_std
    y_val_n = (y_val - y_mean) / y_std

    input_dim = xs.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batches_per_epoch = (len(x_train_n) + args.batch - 1) // args.batch
    arch = f"{hidden1}" if hidden2 <= 0 else f"{hidden1}x{hidden2}"
    kind = "int8" if int8 else "float"
    log(
        f"  arch={arch} {kind}  input={input_dim}  device={device}  "
        f"train={len(x_train_n):,}  val={len(x_val_n):,}  "
        f"batch={args.batch}  ({batches_per_epoch} batches/epoch)  epochs={args.epochs}"
    )

    model = MLP(input_dim, hidden1, hidden2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train_n), torch.from_numpy(y_train_n)),
        batch_size=args.batch,
        shuffle=True,
    )
    val_x = torch.from_numpy(x_val_n).to(device)

    best = {"pos": 1e9, "g": 0.0, "state": None}
    train_t0 = time.perf_counter()
    epoch_times = []

    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.perf_counter()
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = (
                mse(pred[:, :3], yb[:, :3]) * 25
                + mse(pred[:, 3:6], yb[:, 3:6]) * 5
                + mse(pred[:, 6], yb[:, 6]) * 10
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(val_x).cpu().numpy()
        pos, vel, g = metrics(pred_val, y_val_n, y_mean, y_std)
        if pos < best["pos"]:
            best["pos"] = pos
            best["g"] = g
            best["state"] = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        epoch_dt = time.perf_counter() - epoch_t0
        epoch_times.append(epoch_dt)
        should_log = args.verbose or epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0
        if should_log:
            avg_epoch = sum(epoch_times) / len(epoch_times)
            remaining_epochs = args.epochs - epoch
            eta = avg_epoch * remaining_epochs
            elapsed = time.perf_counter() - train_t0
            log(
                f"  ep {epoch:3d}/{args.epochs}  "
                f"pos={pos:.5f}  vel={vel:.5f}  g={g:.4f}  "
                f"epoch={epoch_dt:.1f}s  elapsed={fmt_duration(elapsed)}  eta={fmt_duration(eta)}"
            )

    model.load_state_dict(best["state"])
    w1, b1, w2, b2, w3, b3 = get_weights(model, hidden2)
    fw1, fb1, fw2, fb2, fw3, fb3 = fuse_weights(w1, b1, w2, b2, w3, b3, x_mean, x_std, y_mean, y_std)

    out = Path(out_path)
    if int8:
        export_int8(out, fw1, fb1, fw2, fb2, fw3, fb3, input_dim, hidden1, hidden2)
    else:
        export_float(out, w1, b1, w2, b2, w3, b3, x_mean, x_std, y_mean, y_std, input_dim, hidden1, hidden2)

    total = time.perf_counter() - train_t0
    log(f"  -> {out.name} ({arch} {kind}) pos={best['pos']:.5f} grounded={best['g']:.4f}  ({fmt_duration(total)})")
    return best["pos"], best["g"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train.bin")
    ap.add_argument("--out", default="models/model.bin")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden1", type=int, default=24)
    ap.add_argument("--hidden2", type=int, default=0)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--sweep", action="store_true", help="train int8 speed ladder (h12-h32)")
    ap.add_argument("--sweep-float", action="store_true", help="train float v1 ladder (h12-h32)")
    ap.add_argument("--sweep-stable", action="store_true", help="train larger int8 models for rollout stability")
    ap.add_argument("--verbose", action="store_true", help="log every epoch (default: every --log-every)")
    ap.add_argument("--log-every", type=int, default=5, help="status interval in epochs")
    args = ap.parse_args()

    if args.sweep or args.sweep_float or args.sweep_stable:
        if args.sweep_stable:
            configs = [
                (48, 0, "model_q8_h48_p3.bin", True),
                (64, 0, "model_q8_h64_p3.bin", True),
                (48, 24, "model_q8_h48x24_p3.bin", True),
            ]
            kind = "stable int8 (3^3 patch)"
        elif args.sweep:
            configs = [
                (12, 0, "model_q8_h12_p3.bin", True),
                (16, 0, "model_q8_h16_p3.bin", True),
                (24, 0, "model_q8_h24_p3.bin", True),
                (32, 0, "model_q8_h32_p3.bin", True),
            ]
            kind = "int8"
        else:
            configs = [
                (12, 0, "model_float_h12_p3.bin", False),
                (16, 0, "model_float_h16_p3.bin", False),
                (24, 0, "model_float_h24_p3.bin", False),
                (32, 0, "model_float_h32_p3.bin", False),
            ]
            kind = "float"
        sweep_t0 = time.perf_counter()
        log(f"{kind} sweep on {args.data}  ({args.epochs} epochs x {len(configs)} models)")
        dataset = load_dataset(Path(args.data))
        results = []
        for i, (h1, h2, name, q8) in enumerate(configs, 1):
            log(f"[{i}/{len(configs)}] Training {name} ...")
            pos, g = train_one(args, h1, h2, f"models/{name}", int8=q8, dataset=dataset)
            results.append((name, pos, g))
        log(f"\n=== {kind} sweep summary ({fmt_duration(time.perf_counter() - sweep_t0)} total) ===")
        for name, pos, g in results:
            ok = "OK" if g >= 0.98 else "LOW"
            log(f"  {name:26s}  pos={pos:.5f}  grounded={g:.4f}  [{ok}]")
        return

    train_one(args, args.hidden1, args.hidden2, args.out, int8=args.int8)


if __name__ == "__main__":
    main()
