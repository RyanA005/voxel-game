#!/usr/bin/env python3
"""Train MLP physics model and export models/model.bin for C inference."""

import argparse
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

INPUT_DIM = 741
OUTPUT_DIM = 7
HIDDEN1 = 128
HIDDEN2 = 128
MODEL_MAGIC = 0x214D4C50
DATASET_MAGIC = 0x4B435056


def load_dataset(path: Path):
    data = path.read_bytes()
    magic, version, record_size = struct.unpack_from("<IHH", data, 0)
    assert magic == DATASET_MAGIC, f"bad magic {magic:#x}"
    assert version == 1

    offset = 8
    xs, ys, seeds = [], [], []
    while offset + record_size <= len(data):
        rec = data[offset : offset + record_size]
        offset += record_size

        voxels = np.frombuffer(rec, dtype=np.uint8, count=729, offset=0).astype(np.float32) / 4.0
        off = np.frombuffer(rec, dtype=np.float32, count=3, offset=729)
        vel = np.frombuffer(rec, dtype=np.float32, count=3, offset=741)
        grounded = float(rec[753])
        inp = np.frombuffer(rec, dtype=np.uint8, count=5, offset=754).astype(np.float32)
        delta = np.frombuffer(rec, dtype=np.float32, count=3, offset=763)
        tvel = np.frombuffer(rec, dtype=np.float32, count=3, offset=775)
        tground = float(rec[787])
        seed = struct.unpack_from("<I", rec, 788)[0]

        vel_n = np.array([vel[0] / 6.0, vel[1] / 10.0, vel[2] / 6.0], dtype=np.float32)
        x = np.concatenate([voxels, off, vel_n, [grounded], inp])
        y = np.concatenate([delta, tvel, [tground]])
        xs.append(x)
        ys.append(y)
        seeds.append(seed)

    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32), np.array(seeds, dtype=np.uint32)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN1),
            nn.ReLU(),
            nn.Linear(HIDDEN1, HIDDEN2),
            nn.ReLU(),
            nn.Linear(HIDDEN2, OUTPUT_DIM),
        )

    def forward(self, x):
        return self.net(x)


def split_by_seed(seeds, val_ratio=0.2):
    unique = np.unique(seeds)
    rng = np.random.default_rng(42)
    rng.shuffle(unique)
    n_val = max(1, int(len(unique) * val_ratio))
    val_seeds = set(unique[:n_val].tolist())
    return np.array([s in val_seeds for s in seeds])


def export_model(path: Path, model, x_mean, x_std, y_mean, y_std):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    w1 = state["net.0.weight"].cpu().numpy().astype(np.float32)
    b1 = state["net.0.bias"].cpu().numpy().astype(np.float32)
    w2 = state["net.2.weight"].cpu().numpy().astype(np.float32)
    b2 = state["net.2.bias"].cpu().numpy().astype(np.float32)
    w3 = state["net.4.weight"].cpu().numpy().astype(np.float32)
    b3 = state["net.4.bias"].cpu().numpy().astype(np.float32)

    with path.open("wb") as f:
        f.write(struct.pack("<II", MODEL_MAGIC, 1))
        f.write(struct.pack("<4i", INPUT_DIM, HIDDEN1, HIDDEN2, OUTPUT_DIM))
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


def denorm(pred_n, y_mean, y_std):
    return pred_n * y_std + y_mean


def metrics(pred_n, y_n, y_mean, y_std):
    pred = denorm(pred_n, y_mean, y_std)
    target = denorm(y_n, y_mean, y_std)
    pos_rmse = float(np.sqrt(((pred[:, :3] - target[:, :3]) ** 2).mean()))
    vel_rmse = float(np.sqrt(((pred[:, 3:6] - target[:, 3:6]) ** 2).mean()))
    g_acc = float(((pred[:, 6] >= 0.5) == (target[:, 6] >= 0.5)).mean())
    return pos_rmse, vel_rmse, g_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train.bin")
    ap.add_argument("--out", default="models/model.bin")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    xs, ys, seeds = load_dataset(Path(args.data))
    print(f"Loaded {len(xs)} samples")

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train_n), torch.from_numpy(y_train_n)),
        batch_size=args.batch,
        shuffle=True,
    )
    val_x = torch.from_numpy(x_val_n).to(device)
    val_y = torch.from_numpy(y_val_n).to(device)

    model = MLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    y_mean_t = torch.from_numpy(y_mean.astype(np.float32)).to(device)
    y_std_t = torch.from_numpy(y_std.astype(np.float32)).to(device)

    best_pos = 1e9
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss_pos = mse(pred[:, :3], yb[:, :3]) * 20.0
            loss_vel = mse(pred[:, 3:6], yb[:, 3:6]) * 2.0
            loss_g = mse(pred[:, 6], yb[:, 6]) * 5.0
            loss = loss_pos + loss_vel + loss_g
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(xb)

        model.eval()
        with torch.no_grad():
            pred_val = model(val_x)
            pos_rmse, vel_rmse, g_acc = metrics(
                pred_val.cpu().numpy(), y_val_n, y_mean, y_std
            )
        print(
            f"epoch {epoch:2d}  loss={total/len(x_train):.4f}  "
            f"val_pos={pos_rmse:.5f}  val_vel={vel_rmse:.5f}  val_g={g_acc:.3f}"
        )
        if pos_rmse < best_pos:
            best_pos = pos_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    export_model(Path(args.out), model, x_mean, x_std, y_mean, y_std)
    print(f"Exported {args.out} (best val_pos={best_pos:.5f})")


if __name__ == "__main__":
    main()
