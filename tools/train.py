#!/usr/bin/env python3
"""Train MLP physics model with teacher-forced + closed-loop rollout; export models/*.bin."""

import argparse
import struct
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from physics_sim import (  # noqa: E402
    apply_neural_step,
    copy_state,
    generate_world,
    init_state_from_pos_vel,
    pack_observation,
    player_collides,
)

FULL_PATCH = 9
INPUT_EXTRA = 12
OUTPUT_DIM = 7
POS_BEFORE_OFF = 792
MODEL_MAGIC = 0x214D4C50
DATASET_MAGIC = 0x4B435056


def patch_input_dim(patch_n: int) -> int:
    return patch_n ** 3 + INPUT_EXTRA


def patch_feature_indices(patch_n: int):
    v = patch_n ** 3
    return v + 3 + 3, v + 3 + 3 + 1  # grounded_idx, keys_idx


def crop_voxels(v729: np.ndarray, patch_n: int) -> np.ndarray:
    if patch_n == FULL_PATCH:
        return v729
    pad = (FULL_PATCH - patch_n) // 2
    v = v729.reshape(FULL_PATCH, FULL_PATCH, FULL_PATCH)
    return v[pad : pad + patch_n, pad : pad + patch_n, pad : pad + patch_n].reshape(-1)


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def fmt_eta(seconds: float) -> str:
    if seconds < 0 or not np.isfinite(seconds):
        return "??:??:??"
    return str(timedelta(seconds=int(seconds)))


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)


def load_dataset(path: Path, patch_n: int = FULL_PATCH, max_samples: int | None = None):
    data = path.read_bytes()
    magic, version, record_size = struct.unpack_from("<IHH", data, 0)
    assert magic == DATASET_MAGIC, f"bad magic {magic:#x}"
    assert version in (1, 2), f"unsupported dataset version {version}"
    assert record_size in (792, 804), f"unexpected record_size {record_size}"
    has_pos = record_size == 804

    n_total = (len(data) - 8) // record_size
    if max_samples is not None:
        n_total = min(n_total, max_samples)

    dim = patch_input_dim(patch_n)
    seeds = np.empty(n_total, dtype=np.uint32)
    offset = 8
    for i in range(n_total):
        seeds[i] = struct.unpack_from("<I", data, offset + 788)[0]
        offset += record_size

    val_mask = split_by_seed(seeds)
    n_train = int((~val_mask).sum())
    n_val = int(val_mask.sum())

    x_train = np.empty((n_train, dim), dtype=np.float32)
    y_train = np.empty((n_train, OUTPUT_DIM), dtype=np.float32)
    x_val = np.empty((n_val, dim), dtype=np.float32)
    y_val = np.empty((n_val, OUTPUT_DIM), dtype=np.float32)
    train_seeds = np.empty(n_train, dtype=np.uint32)
    val_seeds = np.empty(n_val, dtype=np.uint32)
    pos_train = np.empty((n_train, 3), dtype=np.float32) if has_pos else None
    pos_val = np.empty((n_val, 3), dtype=np.float32) if has_pos else None
    vel_train = np.empty((n_train, 3), dtype=np.float32)
    vel_val = np.empty((n_val, 3), dtype=np.float32)
    g_train = np.empty(n_train, dtype=np.uint8)
    g_val = np.empty(n_val, dtype=np.uint8)
    inp_train = np.empty((n_train, 5), dtype=np.uint8)
    inp_val = np.empty((n_val, 5), dtype=np.uint8)

    ti = vi = 0
    offset = 8
    for i in range(n_total):
        rec = data[offset : offset + record_size]
        offset += record_size

        voxels = crop_voxels(
            np.frombuffer(rec, dtype=np.uint8, count=729, offset=0).astype(np.float32) / 4.0,
            patch_n,
        )
        off = np.frombuffer(rec, dtype=np.float32, count=3, offset=729)
        vel = np.frombuffer(rec, dtype=np.float32, count=3, offset=741)
        grounded = float(rec[753])
        inp = np.frombuffer(rec, dtype=np.uint8, count=5, offset=754).astype(np.float32)
        delta = np.frombuffer(rec, dtype=np.float32, count=3, offset=763)
        tvel = np.frombuffer(rec, dtype=np.float32, count=3, offset=775)
        tground = float(rec[787])
        seed = seeds[i]

        vel_n = np.array([vel[0] / 6.0, vel[1] / 10.0, vel[2] / 6.0], dtype=np.float32)
        x = np.concatenate([voxels, off, vel_n, [grounded], inp])
        y = np.concatenate([delta, tvel, [tground]])

        if val_mask[i]:
            x_val[vi] = x
            y_val[vi] = y
            val_seeds[vi] = seed
            vel_val[vi] = vel
            g_val[vi] = rec[753]
            inp_val[vi] = np.frombuffer(rec, dtype=np.uint8, count=5, offset=754)
            if has_pos:
                pos_val[vi] = np.frombuffer(rec, dtype=np.float32, count=3, offset=POS_BEFORE_OFF)
            vi += 1
        else:
            x_train[ti] = x
            y_train[ti] = y
            train_seeds[ti] = seed
            vel_train[ti] = vel
            g_train[ti] = rec[753]
            inp_train[ti] = np.frombuffer(rec, dtype=np.uint8, count=5, offset=754)
            if has_pos:
                pos_train[ti] = np.frombuffer(rec, dtype=np.float32, count=3, offset=POS_BEFORE_OFF)
            ti += 1

    meta = {
        "pos_before": pos_train,
        "vel_raw": vel_train,
        "grounded_raw": g_train,
        "inputs_raw": inp_train,
        "seeds": train_seeds,
        "has_pos": has_pos,
    }
    return x_train, y_train, x_val, y_val, train_seeds, val_seeds, meta


def build_episodes(seeds: np.ndarray, max_len: int = 400, min_len: int = 8):
    episodes = []
    start = 0
    for i in range(1, len(seeds)):
        if seeds[i] != seeds[i - 1] or (i - start) >= max_len:
            if i - start >= min_len:
                episodes.append((start, i))
            start = i
    if len(seeds) - start >= min_len:
        episodes.append((start, len(seeds)))
    return episodes


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden1: int, hidden2: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, OUTPUT_DIM),
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


def export_model(path: Path, model: MLP, x_mean, x_std, y_mean, y_std):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    h1, h2 = model.hidden1, model.hidden2
    in_dim = model.input_dim
    w1 = state["net.0.weight"].cpu().numpy().astype(np.float32)
    b1 = state["net.0.bias"].cpu().numpy().astype(np.float32)
    w2 = state["net.2.weight"].cpu().numpy().astype(np.float32)
    b2 = state["net.2.bias"].cpu().numpy().astype(np.float32)
    w3 = state["net.4.weight"].cpu().numpy().astype(np.float32)
    b3 = state["net.4.bias"].cpu().numpy().astype(np.float32)

    with path.open("wb") as f:
        f.write(struct.pack("<II", MODEL_MAGIC, 1))
        f.write(struct.pack("<4i", in_dim, h1, h2, OUTPUT_DIM))
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


def is_idle_grounded_x(x: np.ndarray, patch_n: int) -> np.ndarray:
    grounded_idx, keys_idx = patch_feature_indices(patch_n)
    grounded = x[..., grounded_idx] > 0.5
    keys_off = x[..., keys_idx : keys_idx + 5].sum(axis=-1) == 0
    return grounded & keys_off


def is_airborne_x(x: np.ndarray, patch_n: int) -> np.ndarray:
    grounded_idx, _ = patch_feature_indices(patch_n)
    return x[..., grounded_idx] <= 0.5


def inp_dict(row: np.ndarray) -> dict:
    return {
        "forward": bool(row[0]),
        "back": bool(row[1]),
        "left": bool(row[2]),
        "right": bool(row[3]),
        "jump": bool(row[4]),
    }


def norm_obs(obs: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray) -> np.ndarray:
    return (obs - x_mean) / x_std


def step_loss(pred_n, y_n, mse_none, weights=None):
    loss_pos = mse_none(pred_n[:, :3], y_n[:, :3]).mean(dim=1) * 20.0
    loss_vel = mse_none(pred_n[:, 3:6], y_n[:, 3:6]).mean(dim=1) * 2.0
    loss_g = mse_none(pred_n[:, 6], y_n[:, 6]) * 5.0
    per = loss_pos + loss_vel + loss_g
    if weights is not None:
        return (per * weights).mean()
    return per.mean()


def sample_rollout_batch(x_n, y_n, episodes, batch_size, rollout_steps, rng):
    B = batch_size
    K = rollout_steps
    x_batch = np.zeros((B, K, x_n.shape[1]), dtype=np.float32)
    y_batch = np.zeros((B, K, y_n.shape[1]), dtype=np.float32)

    for b in range(B):
        s, e = episodes[rng.integers(0, len(episodes))]
        t0 = rng.integers(s, e - K)
        x_batch[b] = x_n[t0 : t0 + K]
        y_batch[b] = y_n[t0 : t0 + K]
    return x_batch, y_batch


def closed_loop_loss(
    model,
    meta,
    y_train,
    train_episodes,
    batch_size,
    rollout_steps,
    x_mean,
    x_std,
    y_mean,
    y_std,
    device,
    patch_n,
    rng,
    idle_weight,
    airborne_weight,
    tunnel_penalty,
    mse_none,
):
    pos_before = meta["pos_before"]
    vel_raw = meta["vel_raw"]
    grounded_raw = meta["grounded_raw"]
    inputs_raw = meta["inputs_raw"]
    seeds = meta["seeds"]

    loss = torch.tensor(0.0, device=device)

    for _ in range(batch_size):
        s, e = train_episodes[rng.integers(0, len(train_episodes))]
        t0 = int(rng.integers(s, e - rollout_steps))
        world, _ = generate_world(int(seeds[t0]))
        state = init_state_from_pos_vel(pos_before[t0], vel_raw[t0], grounded_raw[t0])

        for k in range(rollout_steps):
            idx = t0 + k
            inp = inp_dict(inputs_raw[idx])
            obs = pack_observation(world, state["pos"], state["vel"], state["grounded"], inp)
            obs_n = norm_obs(obs, x_mean, x_std)
            x_t = torch.from_numpy(obs_n).unsqueeze(0).to(device)
            pred_n = model(x_t)[0]
            y_t = torch.from_numpy(y_train[idx]).to(device)
            idle = float(is_idle_grounded_x(obs_n, patch_n))
            air = float(is_airborne_x(obs_n, patch_n))
            w = 1.0 + idle * (idle_weight - 1.0) + air * (airborne_weight - 1.0)
            loss = loss + step_loss(
                pred_n.unsqueeze(0), y_t.unsqueeze(0), mse_none, torch.tensor([w], device=device)
            )

            pred = pred_n.detach().cpu().numpy() * y_std + y_mean
            apply_neural_step(state, pred)
            if player_collides(world, state["pos"]) and not state["grounded"]:
                loss = loss + tunnel_penalty

    return loss / batch_size


def eval_one_step(model, x_n, y_n, y_mean, y_std, device, batch=2048):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x_n), batch):
            xb = torch.from_numpy(x_n[i : i + batch]).to(device)
            preds.append(model(xb).cpu().numpy())
    pred_n = np.concatenate(preds, axis=0)
    pred = denorm(pred_n, y_mean, y_std)
    target = denorm(y_n, y_mean, y_std)
    pos_rmse = float(np.sqrt(((pred[:, :3] - target[:, :3]) ** 2).mean()))
    vel_rmse = float(np.sqrt(((pred[:, 3:6] - target[:, 3:6]) ** 2).mean()))
    g_acc = float(((pred[:, 6] >= 0.5) == (target[:, 6] >= 0.5)).mean())
    return pos_rmse, vel_rmse, g_acc


def eval_rollout(model, x_n, y_n, y_mean, y_std, episodes, rollout_steps, device, samples=256, batch=64):
    if rollout_steps < 2 or len(episodes) == 0:
        return 0.0, 0.0

    model.eval()
    rng = np.random.default_rng(123)
    cum_err = []
    vel_err = []

    with torch.no_grad():
        for _ in range(0, samples, batch):
            cur = min(batch, samples - len(cum_err))
            x_batch, y_batch = sample_rollout_batch(x_n, y_n, episodes, cur, rollout_steps, rng)
            x_t = torch.from_numpy(x_batch).to(device)
            y_t = torch.from_numpy(y_batch)

            preds = []
            for k in range(rollout_steps):
                preds.append(model(x_t[:, k, :]))
            pred_n = torch.stack(preds, dim=1)
            pred = pred_n.cpu().numpy() * y_std + y_mean
            target = y_t.numpy() * y_std + y_mean

            d_pred = pred[:, :, :3].cumsum(axis=1)
            d_true = target[:, :, :3].cumsum(axis=1)
            cum_err.append(np.sqrt(((d_pred - d_true) ** 2).mean(axis=2)).mean())

            vel_err.append(np.sqrt(((pred[:, :, 3:6] - target[:, :, 3:6]) ** 2).mean(axis=2)).mean())

    return float(np.mean(cum_err)), float(np.mean(vel_err))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train.bin")
    ap.add_argument("--out", default="models/model.bin")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--batches-per-epoch", type=int, default=800)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--hidden1", type=int, default=256)
    ap.add_argument("--hidden2", type=int, default=256)
    ap.add_argument("--rollout-steps", type=int, default=8)
    ap.add_argument("--rollout-weight", type=float, default=8.0)
    ap.add_argument("--patch", type=int, default=9, help="voxel patch size n (n³ voxels, 2–9)")
    ap.add_argument("--idle-weight", type=float, default=4.0, help="loss multiplier for idle+grounded samples")
    ap.add_argument("--airborne-weight", type=float, default=1.0, help="loss multiplier for airborne samples")
    ap.add_argument("--closed-loop-weight", type=float, default=0.0, help="closed-loop rollout loss scale (0=off)")
    ap.add_argument("--tunnel-penalty", type=float, default=25.0, help="penalty per tunnel step in closed-loop")
    ap.add_argument("--cl-batch", type=int, default=48, help="closed-loop micro-batch size")
    ap.add_argument("--max-samples", type=int, default=0, help="cap loaded samples (0 = all)")
    ap.add_argument("--no-rollout", action="store_true")
    ap.add_argument("--no-closed-loop", action="store_true")
    args = ap.parse_args()

    patch_n = max(2, min(9, args.patch))
    input_dim = patch_input_dim(patch_n)
    max_samples = args.max_samples if args.max_samples > 0 else None

    t_start = time.time()
    log(f"Loading dataset {args.data}  patch={patch_n}³  input_dim={input_dim}")
    x_train, y_train, x_val, y_val, train_seeds, val_seeds_arr, train_meta = load_dataset(
        Path(args.data), patch_n, max_samples=max_samples
    )
    n_total = len(x_train) + len(x_val)
    log(f"Loaded {n_total:,} samples (dataset v{2 if train_meta['has_pos'] else 1})")
    idle_frac = is_idle_grounded_x(x_train, patch_n).mean()
    air_frac = is_airborne_x(x_train, patch_n).mean()
    log(f"Idle+grounded: {idle_frac:.1%}  Airborne obs: {air_frac:.1%}")

    train_episodes = build_episodes(train_seeds)
    val_episodes = build_episodes(val_seeds_arr)
    log(f"Train episodes: {len(train_episodes):,}  Val episodes: {len(val_episodes):,}")

    x_mean = x_train.mean(axis=0, dtype=np.float32)
    x_std = x_train.std(axis=0, dtype=np.float32)
    x_std[x_std < 1e-6] = 1.0
    y_mean = y_train.mean(axis=0, dtype=np.float32)
    y_std = y_train.std(axis=0, dtype=np.float32)
    y_std[y_std < 1e-6] = 1.0

    np.subtract(x_train, x_mean, out=x_train)
    np.divide(x_train, x_std, out=x_train)
    np.subtract(y_train, y_mean, out=y_train)
    np.divide(y_train, y_std, out=y_train)
    np.subtract(x_val, x_mean, out=x_val)
    np.divide(x_val, x_std, out=x_val)
    np.subtract(y_val, y_mean, out=y_val)
    np.divide(y_val, y_std, out=y_val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_rollout = not args.no_rollout and args.rollout_steps >= 2
    use_cl = (
        not args.no_closed_loop
        and args.closed_loop_weight > 0
        and train_meta["has_pos"]
        and patch_n == FULL_PATCH
    )
    log(
        f"Device: {device}  Patch: {patch_n}³  Model: {args.hidden1}x{args.hidden2}  "
        f"Rollout: {args.rollout_steps if use_rollout else 'off'}  "
        f"Closed-loop: {'on' if use_cl else 'off'}  "
        f"Idle weight: {args.idle_weight}x  Airborne: {args.airborne_weight}x  "
        f"Out: {args.out}"
    )

    model = MLP(input_dim, args.hidden1, args.hidden2).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Parameters: {n_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse_none = nn.MSELoss(reduction="none")
    mse_mean = nn.MSELoss(reduction="mean")
    rng = np.random.default_rng(42)

    y_std_t = torch.from_numpy(y_std.astype(np.float32)).to(device)
    y_mean_t = torch.from_numpy(y_mean.astype(np.float32)).to(device)

    best_score = 1e9
    best_state = None

    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.time()
        model.train()
        running = 0.0
        n_samples = 0

        for batch_i in range(1, args.batches_per_epoch + 1):
            batch_t0 = time.time()
            do_cl = use_cl and (batch_i % 2 == 0)

            if do_cl:
                cl_loss = closed_loop_loss(
                    model,
                    train_meta,
                    y_train,
                    train_episodes,
                    args.cl_batch,
                    args.rollout_steps,
                    x_mean,
                    x_std,
                    y_mean,
                    y_std,
                    device,
                    patch_n,
                    rng,
                    args.idle_weight,
                    args.airborne_weight,
                    args.tunnel_penalty,
                    mse_none,
                )
                loss = cl_loss * args.closed_loop_weight
            elif use_rollout:
                x_batch, y_batch = sample_rollout_batch(
                    x_train, y_train, train_episodes, args.batch, args.rollout_steps, rng
                )
                x_t = torch.from_numpy(x_batch).to(device)
                y_t = torch.from_numpy(y_batch).to(device)

                preds = []
                loss = torch.tensor(0.0, device=device)
                for k in range(args.rollout_steps):
                    pred_k = model(x_t[:, k, :])
                    preds.append(pred_k)
                    xk = x_batch[:, k, :]
                    idle = torch.from_numpy(is_idle_grounded_x(xk, patch_n).astype(np.float32)).to(device)
                    air = torch.from_numpy(is_airborne_x(xk, patch_n).astype(np.float32)).to(device)
                    w = 1.0 + idle * (args.idle_weight - 1.0) + air * (args.airborne_weight - 1.0)
                    loss = loss + step_loss(pred_k, y_t[:, k, :], mse_none, w)

                pred_stack = torch.stack(preds, dim=1)
                if args.rollout_weight > 0:
                    cum_pred = pred_stack[:, :, :3].cumsum(dim=1)
                    cum_true = y_t[:, :, :3].cumsum(dim=1)
                    loss = loss + mse_mean(cum_pred, cum_true) * args.rollout_weight
                    vel_pred = pred_stack[:, :, 3:6]
                    vel_true = y_t[:, :, 3:6]
                    loss = loss + mse_mean(vel_pred, vel_true) * (args.rollout_weight * 0.5)
            else:
                idx = rng.integers(0, len(x_train), size=args.batch)
                x_t = torch.from_numpy(x_train[idx]).to(device)
                y_t = torch.from_numpy(y_train[idx]).to(device)
                pred = model(x_t)
                idle = torch.from_numpy(is_idle_grounded_x(x_train[idx], patch_n).astype(np.float32)).to(device)
                air = torch.from_numpy(is_airborne_x(x_train[idx], patch_n).astype(np.float32)).to(device)
                w = 1.0 + idle * (args.idle_weight - 1.0) + air * (args.airborne_weight - 1.0)
                loss = step_loss(pred, y_t, mse_none, w)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running += loss.item() * args.batch
            n_samples += args.batch

            if batch_i % max(1, args.batches_per_epoch // 5) == 0 or batch_i == args.batches_per_epoch:
                elapsed = time.time() - t_start
                epoch_elapsed = time.time() - epoch_t0
                batches_left_epoch = args.batches_per_epoch - batch_i
                batches_left_total = batches_left_epoch + (args.epochs - epoch) * args.batches_per_epoch
                sec_per_batch = epoch_elapsed / batch_i
                eta_total = sec_per_batch * batches_left_total
                log(
                    f"epoch {epoch}/{args.epochs}  batch {batch_i}/{args.batches_per_epoch}  "
                    f"loss={running/n_samples:.4f}  "
                    f"batch={time.time()-batch_t0:.2f}s  "
                    f"elapsed={fmt_eta(elapsed)}  eta={fmt_eta(eta_total)}"
                )

        pos_rmse, vel_rmse, g_acc = eval_one_step(model, x_val, y_val, y_mean, y_std, device)
        rollout_pos, rollout_vel = eval_rollout(
            model, x_val, y_val, y_mean, y_std, val_episodes, args.rollout_steps, device
        )

        score = pos_rmse + rollout_pos * 0.5
        log(
            f"epoch {epoch}/{args.epochs} DONE  "
            f"train_loss={running/n_samples:.4f}  "
            f"val_pos={pos_rmse:.5f}  val_vel={vel_rmse:.5f}  val_g={g_acc:.3f}  "
            f"val_rollout_pos={rollout_pos:.5f}  val_rollout_vel={rollout_vel:.5f}  "
            f"epoch_time={fmt_eta(time.time()-epoch_t0)}  "
            f"elapsed={fmt_eta(time.time()-t_start)}"
        )

        if score < best_score:
            best_score = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            log(f"  new best score={best_score:.5f} (val_pos + 0.5*rollout_pos)")

    if best_state:
        model.load_state_dict(best_state)
    export_model(Path(args.out), model, x_mean, x_std, y_mean, y_std)
    log(f"Exported {args.out}  total_time={fmt_eta(time.time()-t_start)}  best_score={best_score:.5f}")


if __name__ == "__main__":
    main()
