#!/usr/bin/env python3
"""Export FP32 models to quantized v2 .bin files (int8/int4/fp16)."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_MAGIC = 0x214D4C50
MODEL_VERSION_FP32 = 1
MODEL_VERSION_QUANT = 2
QUANT_INT8_ROW = 1
QUANT_INT8_LAYER = 2
QUANT_INT4_ROW = 3
QUANT_FP16 = 4
OUTPUT_DIM = 7

SCHEMES = {
    "int8_row": QUANT_INT8_ROW,
    "int8_layer": QUANT_INT8_LAYER,
    "int4_row": QUANT_INT4_ROW,
    "fp16": QUANT_FP16,
}


def load_fp32_model(path: Path) -> dict:
    data = path.read_bytes()
    off = 0
    magic, version = struct.unpack_from("<II", data, off)
    off += 8
    if magic != MODEL_MAGIC or version != MODEL_VERSION_FP32:
        raise ValueError(f"{path}: expected FP32 v1 model")

    in_dim, h1, h2, out_dim = struct.unpack_from("<4i", data, off)
    off += 16

    def take(n: int) -> np.ndarray:
        nonlocal off
        arr = np.frombuffer(data, dtype=np.float32, count=n, offset=off).copy()
        off += 4 * n
        return arr

    x_mean = take(in_dim)
    x_std = take(in_dim)
    y_mean = take(out_dim)
    y_std = take(out_dim)
    w1 = take(h1 * in_dim).reshape(h1, in_dim)
    b1 = take(h1)
    w2 = take(h2 * h1).reshape(h2, h1)
    b2 = take(h2)
    w3 = take(out_dim * h2).reshape(out_dim, h2)
    b3 = take(out_dim)

    return {
        "in_dim": in_dim,
        "h1": h1,
        "h2": h2,
        "out_dim": out_dim,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
        "w3": w3,
        "b3": b3,
    }


def quantize_int8_row(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scales = np.max(np.abs(w), axis=1)
    scales = np.maximum(scales / 127.0, 1e-8).astype(np.float32)
    w_q = np.clip(np.round(w / scales[:, None]), -127, 127).astype(np.int8)
    return w_q, scales


def quantize_int8_layer(w: np.ndarray) -> tuple[np.ndarray, float]:
    scale = float(max(np.max(np.abs(w)) / 127.0, 1e-8))
    w_q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    return w_q, scale


def quantize_int4_row(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scales = np.max(np.abs(w), axis=1)
    scales = np.maximum(scales / 7.0, 1e-8).astype(np.float32)
    w_q = np.clip(np.round(w / scales[:, None]), -8, 7).astype(np.int8)
    out_n, in_n = w_q.shape
    row_bytes = (in_n + 1) // 2
    packed = np.zeros((out_n, row_bytes), dtype=np.uint8)
    for o in range(out_n):
        for i in range(in_n):
            v = int(w_q[o, i]) & 0xF
            if i & 1:
                packed[o, i // 2] |= (v << 4)
            else:
                packed[o, i // 2] = v
    return packed.reshape(-1), scales


def write_int8_row_layer(f, w: np.ndarray, b: np.ndarray) -> None:
    w_q, scales = quantize_int8_row(w)
    f.write(scales.astype(np.float32).tobytes())
    f.write(w_q.tobytes())
    f.write(b.astype(np.float32).tobytes())


def write_int8_layer_layer(f, w: np.ndarray, b: np.ndarray) -> None:
    w_q, scale = quantize_int8_layer(w)
    f.write(struct.pack("<f", scale))
    f.write(w_q.tobytes())
    f.write(b.astype(np.float32).tobytes())


def write_int4_row_layer(f, w: np.ndarray, b: np.ndarray) -> None:
    packed, scales = quantize_int4_row(w)
    f.write(scales.astype(np.float32).tobytes())
    f.write(packed.tobytes())
    f.write(b.astype(np.float32).tobytes())


def write_fp16_layer(f, w: np.ndarray, b: np.ndarray) -> None:
    f.write(w.astype(np.float16).tobytes())
    f.write(b.astype(np.float32).tobytes())


def export_quantized(model: dict, out_path: Path, scheme: str) -> None:
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme: {scheme}")

    quant_type = SCHEMES[scheme]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("wb") as f:
        f.write(struct.pack("<II", MODEL_MAGIC, MODEL_VERSION_QUANT))
        f.write(struct.pack("<I", quant_type))
        f.write(
            struct.pack(
                "<4i",
                model["in_dim"],
                model["h1"],
                model["h2"],
                model["out_dim"],
            )
        )
        f.write(model["x_mean"].astype(np.float32).tobytes())
        f.write(model["x_std"].astype(np.float32).tobytes())
        f.write(model["y_mean"].astype(np.float32).tobytes())
        f.write(model["y_std"].astype(np.float32).tobytes())

        layers = [
            (model["w1"], model["b1"]),
            (model["w2"], model["b2"]),
            (model["w3"], model["b3"]),
        ]

        for w, b in layers:
            if scheme == "int8_row":
                write_int8_row_layer(f, w, b)
            elif scheme == "int8_layer":
                write_int8_layer_layer(f, w, b)
            elif scheme == "int4_row":
                write_int4_row_layer(f, w, b)
            elif scheme == "fp16":
                write_fp16_layer(f, w, b)


def default_out_path(src: Path, scheme: str) -> Path:
    rel = src.relative_to(ROOT / "models")
    return ROOT / "models" / "quant" / scheme / rel


# Diverse set: fidelity anchor, speed floor, full 9³, tiny 2³ patch, legacy 128×128.
DEFAULT_MODELS = [
    "patch3/256_rollout8_idle4.bin",
    "patch3/64_rollout4_fast.bin",
    "model_rollout_v2.bin",
    "patch_sweep/patch_2.bin",
    "model.bin",
]


def main():
    ap = argparse.ArgumentParser(description="Quantize FP32 models to v2 formats")
    ap.add_argument("--in", dest="inputs", nargs="*", help="Source FP32 model(s)")
    ap.add_argument("--all-defaults", action="store_true", help="Quantize the 5 diverse models")
    ap.add_argument(
        "--scheme",
        choices=list(SCHEMES),
        action="append",
        help="Quant scheme (repeatable; default: all)",
    )
    ap.add_argument("--out", help="Output path (single input only)")
    args = ap.parse_args()

    schemes = args.scheme or list(SCHEMES)
    if args.all_defaults:
        inputs = [ROOT / "models" / p for p in DEFAULT_MODELS]
    elif args.inputs:
        inputs = [Path(p) for p in args.inputs]
    else:
        ap.error("provide --in MODEL or --all-defaults")

    if args.out and len(inputs) != 1:
        ap.error("--out requires exactly one input model")

    for src in inputs:
        if not src.is_absolute():
            src = ROOT / src
        model = load_fp32_model(src)
        for scheme in schemes:
            out = Path(args.out) if args.out else default_out_path(src, scheme)
            export_quantized(model, out, scheme)
            size_kb = out.stat().st_size / 1024
            print(f"  {out.relative_to(ROOT)}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
