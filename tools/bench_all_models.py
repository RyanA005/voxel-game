#!/usr/bin/env python3
"""Benchmark all models vs analytic teacher; print ranked table for progress-log.md."""

import re
import struct
import subprocess
import sys
from pathlib import Path

# Windows consoles often lack UTF-8; keep output ASCII-safe.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "build" / "voxel_parkour.exe"
MODEL_MAGIC = 0x214D4C50

SKIP = {"patch_9_test.bin", "best.bin"}
ANCHOR = "256_rollout8_idle4.bin"

# Pre Phase 12 C inference timings (Phase 11 snapshot).
PRE_OPT_US = {
    "256_rollout8_idle4.bin": 72,
    "256_rollout8_30ep.bin": 72,
    "64_rollout4_fast.bin": 5.7,
    "model_rollout_v2.bin": 268,
    "patch_2.bin": 16,
    "model_rollout.bin": 252,
    "128_rollout4_30ep.bin": 18,
    "model.bin": 145,
    "patch_5.bin": 29,
    "patch_9.bin": 118,
    "patch_4.bin": 22,
    "patch_7.bin": 61,
    "patch_6.bin": 42,
    "patch_8.bin": 85,
    "patch_3.bin": 18,
}

PHASE12_START = "<!-- bench-phase12-start -->"
PHASE12_END = "<!-- bench-phase12-end -->"


def patch_from_input_dim(d: int) -> int:
    v = d - 12
    for n in range(2, 10):
        if n ** 3 == v:
            return n
    return 9


def read_model_info(path: Path) -> dict:
    data = path.read_bytes()
    magic, version = struct.unpack_from("<II", data, 0)
    if magic != MODEL_MAGIC:
        return {"input_dim": 0, "h1": 0, "h2": 0, "patch": "?"}
    in_dim, h1, h2, out_dim = struct.unpack_from("<4i", data, 8)
    p = patch_from_input_dim(in_dim)
    return {"input_dim": in_dim, "h1": h1, "h2": h2, "patch": f"{p}\u00b3"}


def bench(path: Path) -> dict:
    text = subprocess.check_output([str(EXE), "--bench", str(path)], cwd=ROOT, text=True)

    def grab(pat, default=0.0):
        m = re.search(pat, text)
        return float(m.group(1)) if m else default

    ta = re.search(r"Tunnel \(analytic\):\s+(\d+) / (\d+)", text)
    tn = re.search(r"Tunnel \(neural\):\s+(\d+) / (\d+)", text)
    return {
        "pos": grab(r"Mean position error:\s+([\d.]+)"),
        "vel": grab(r"Mean velocity error:\s+([\d.]+)"),
        "grounded_mm": grab(r"Grounded mismatch:\s+([\d.]+)"),
        "tunnel_a": int(ta.group(1)) if ta else 0,
        "tunnel_n": int(tn.group(1)) if tn else 0,
        "steps": int(tn.group(2)) if tn else 0,
        "analytic_us": grab(r"Avg analytic step:\s+([\d.]+)"),
        "neural_us": grab(r"Avg neural step:\s+([\d.]+)"),
        "forward_us": grab(r"Forward-only \(10k avg\):\s+([\d.]+)"),
    }


def collect_models() -> list[Path]:
    models = []
    for p in sorted(ROOT.glob("models/**/*.bin")):
        if p.name in SKIP:
            continue
        models.append(p)
    return models


def short_label(path: Path) -> str:
    rel = path.relative_to(ROOT / "models").as_posix()
    return rel.replace(".bin", "")


def fmt_speedup(before: float | None, after: float) -> str:
    if before and before > 0 and after > 0:
        return f"{before / after:.1f}\u00d7"
    return "-"


def fmt_vs_v2(v2_us: float | None, neural_us: float) -> str:
    if v2_us and neural_us > 0:
        return f"{v2_us / neural_us:.1f}\u00d7 faster"
    return "-"


def fmt_analytic_ratio(step_us: float, analytic_us: float) -> str:
    if analytic_us <= 0 or step_us <= 0:
        return "-"
    ratio = step_us / analytic_us
    if ratio >= 10:
        return f"~{ratio:.0f}\u00d7"
    return f"~{ratio:.1f}\u00d7"


def build_phase12_table(rows: list[dict], analytic_us: float, v2_us: float | None) -> str:
    lines = [
        PHASE12_START,
        "| Model | Patch | MLP | Pos err | Vel err | Grounded mm% | Tunnel (N) | Pre-opt \u00b5s | Post-opt \u00b5s | \u00d7 analytic | Inf speedup | vs 9v2 |",
        "|-------|-------|-----|---------|---------|--------------|------------|-------------|--------------|------------|-------------|--------|",
        f"| *(analytic teacher)* | \u2014 | \u2014 | 0 | 0 | 0 | 0 | \u2014 | **~{analytic_us:.1f}** | **1\u00d7** | \u2014 | oracle |",
    ]

    for r in rows:
        arch = f"{r['info']['h1']}\u00d7{r['info']['h2']}"
        pre = PRE_OPT_US.get(r["path"].name)
        pre_s = f"{pre:.1f}" if pre is not None else "\u2014"
        post = r["neural_us"]
        inf_sp = fmt_speedup(pre, post)
        vs_v2 = fmt_vs_v2(v2_us, post)
        x_analytic = fmt_analytic_ratio(post, analytic_us)
        mark = " \u2b50" if r["path"].name == ANCHOR else ""
        star = "**" if r["path"].name == ANCHOR else ""
        pos_s = f"{star}{r['pos']:.3f}{star}" if r["path"].name == ANCHOR else f"{r['pos']:.3f}"
        post_s = f"**{post:.1f}**" if r["path"].name in {ANCHOR, "64_rollout4_fast.bin"} else f"{post:.1f}"
        inf_sp_s = f"**{inf_sp}**" if r["path"].name in {ANCHOR, "64_rollout4_fast.bin"} else inf_sp
        lines.append(
            f"| `{short_label(r['path'])}`{mark} | {r['info']['patch']} | {arch} | {pos_s} | {r['vel']:.3f} | "
            f"{r['grounded_mm']:.1f} | {r['tunnel_n']} | {pre_s} | {post_s} | {x_analytic} | {inf_sp_s} | {vs_v2} |"
        )

    lines.append(PHASE12_END)
    return "\n".join(lines)


def update_progress_log(rows: list[dict], analytic_us: float, v2_us: float | None) -> None:
    log_path = ROOT / "progress-log.md"
    if not log_path.exists():
        return
    text = log_path.read_text(encoding="utf-8")
    start = text.find(PHASE12_START)
    end = text.find(PHASE12_END)
    if start == -1 or end == -1:
        print(f"\nMissing {PHASE12_START} markers in progress-log.md", flush=True)
        return
    end += len(PHASE12_END)
    new_block = build_phase12_table(rows, analytic_us, v2_us)
    text = text[:start] + new_block + text[end:]
    log_path.write_text(text, encoding="utf-8")
    print(f"\nUpdated {log_path} Phase 12 table.", flush=True)


def main():
    if not EXE.exists():
        print(f"Missing {EXE}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for path in collect_models():
        info = read_model_info(path)
        stats = bench(path)
        rows.append({"path": path, "info": info, **stats})
        pre = PRE_OPT_US.get(path.name)
        pre_s = f" pre={pre}" if pre else ""
        print(
            f"  {short_label(path)}: pos={stats['pos']:.4f}  us={stats['neural_us']:.1f}{pre_s}",
            flush=True,
        )

    analytic_us = rows[0]["analytic_us"] if rows else 0.2
    v2_us = next((r["neural_us"] for r in rows if r["path"].name == "model_rollout_v2.bin"), None)

    rows.sort(key=lambda r: r["pos"])
    best_fidelity = rows[0]
    fastest = min(rows, key=lambda r: r["neural_us"])

    lines = [
        "# Model Comparison vs Analytic Physics (post Phase 12)",
        "",
        "All models benchmarked with `--bench`: 50 episodes x up to 300 steps, same inputs to analytic and neural.",
        "**Analytic is the oracle** -- position/velocity errors are neural drift from teacher trajectories.",
        "No analytic fallback at runtime.",
        "",
        f"Analytic physics_step: ~{analytic_us:.1f} us (1x baseline; neural rows show x analytic ratio).",
        "",
        "| Model | Patch | MLP | Pos err | Vel err | Grounded mm% | Tunnel (N) | Pre-opt us | Post-opt us | x analytic | Inf speedup | vs 9v2 |",
        "|-------|-------|-----|---------|---------|--------------|------------|------------|-------------|------------|-------------|--------|",
        f"| *(analytic teacher)* | -- | -- | 0 | 0 | 0 | 0 | -- | ~{analytic_us:.1f} | 1x | -- | oracle |",
    ]

    for r in rows:
        arch = f"{r['info']['h1']}x{r['info']['h2']}"
        pre = PRE_OPT_US.get(r["path"].name)
        pre_s = f"{pre:.1f}" if pre is not None else "-"
        post = r["neural_us"]
        inf_sp = fmt_speedup(pre, post)
        vs_v2 = fmt_vs_v2(v2_us, post)
        x_analytic = fmt_analytic_ratio(post, analytic_us).replace("\u00d7", "x")
        mark = " *" if r["path"].name == ANCHOR else ""
        lines.append(
            f"| `{short_label(r['path'])}`{mark} | {r['info']['patch']} | {arch} | {r['pos']:.4f} | {r['vel']:.3f} | "
            f"{r['grounded_mm']:.1f} | {r['tunnel_n']} | {pre_s} | {post:.1f} | {x_analytic} | {inf_sp} | {vs_v2} |"
        )

    lines += [
        "",
        "Highlights:",
        f"- Best fidelity: `{short_label(best_fidelity['path'])}` -- pos err {best_fidelity['pos']:.4f}",
        f"- Fastest inference: `{short_label(fastest['path'])}` -- {fastest['neural_us']:.1f} us/step",
        f"- Fidelity anchor: patch3/{ANCHOR} (marked *)",
        "",
        "Regenerate: python tools/bench_all_models.py (updates Phase 12 in progress-log.md)",
    ]
    print("\n".join(lines))

    update_progress_log(rows, analytic_us, v2_us)


if __name__ == "__main__":
    main()
