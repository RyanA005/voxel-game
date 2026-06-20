#!/usr/bin/env python3
"""Quantize diverse models and benchmark fidelity + speed vs FP32."""

from __future__ import annotations

import re
import subprocess
import sys
import argparse
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "build" / "voxel_parkour.exe"
QUANT_DIR = ROOT / "models" / "quant"

PHASE13_START = "<!-- bench-phase13-start -->"
PHASE13_END = "<!-- bench-phase13-end -->"

DEFAULT_MODELS = [
    ("patch3/256_rollout8_idle4.bin", "3\u00b3 anchor 256\u00d7256"),
    ("patch3/64_rollout4_fast.bin", "3\u00b3 speed 64\u00d764"),
    ("model_rollout_v2.bin", "9\u00b3 full 256\u00d7256"),
    ("patch_sweep/patch_2.bin", "2\u00b3 sweep 128\u00d7128"),
    ("model.bin", "9\u00b3 legacy 128\u00d7128"),
]

SCHEMES = ["fp32", "int8_row", "int8_layer", "int4_row", "fp16"]


def model_path(scheme: str, rel: str) -> Path:
    if scheme == "fp32":
        return ROOT / "models" / rel
    return QUANT_DIR / scheme / rel


def bench(path: Path) -> dict:
    text = subprocess.check_output([str(EXE), "--bench", str(path)], cwd=ROOT, text=True)

    def grab(pat, default=0.0):
        m = re.search(pat, text)
        return float(m.group(1)) if m else default

    return {
        "pos": grab(r"Mean position error:\s+([\d.]+)"),
        "vel": grab(r"Mean velocity error:\s+([\d.]+)"),
        "grounded_mm": grab(r"Grounded mismatch:\s+([\d.]+)"),
        "tunnel_n": int(re.search(r"Tunnel \(neural\):\s+(\d+)", text).group(1)),
        "analytic_us": grab(r"Avg analytic step:\s+([\d.]+)"),
        "neural_us": grab(r"Avg neural step:\s+([\d.]+)"),
        "forward_us": grab(r"Forward-only \(10k avg\):\s+([\d.]+)"),
    }


def file_kb(path: Path) -> float:
    return path.stat().st_size / 1024.0


def fmt_x_analytic(step_us: float, analytic_us: float) -> str:
    if analytic_us <= 0 or step_us <= 0:
        return "-"
    ratio = step_us / analytic_us
    return f"~{ratio:.0f}\u00d7" if ratio >= 10 else f"~{ratio:.1f}\u00d7"


def pos_delta(base: float, cur: float) -> str:
    d = cur - base
    if abs(d) < 0.0005:
        return "0"
    sign = "+" if d > 0 else ""
    return f"{sign}{d:.3f}"


def run_quantize() -> None:
    cmd = [sys.executable, str(ROOT / "tools" / "quantize_models.py"), "--all-defaults"]
    print("Quantizing 5 models x 4 schemes...", flush=True)
    subprocess.check_call(cmd, cwd=ROOT)


def build_rows() -> list[dict]:
    rows = []
    for rel, label in DEFAULT_MODELS:
        fp32_path = model_path("fp32", rel)
        fp32_stats = bench(fp32_path)
        fp32_us = fp32_stats["neural_us"]
        fp32_pos = fp32_stats["pos"]
        analytic_us = fp32_stats["analytic_us"]

        for scheme in SCHEMES:
            path = model_path(scheme, rel)
            if not path.exists():
                print(f"  skip missing {path}", flush=True)
                continue
            stats = fp32_stats if scheme == "fp32" else bench(path)
            rows.append(
                {
                    "label": label,
                    "rel": rel.replace(".bin", ""),
                    "scheme": scheme,
                    "path": path,
                    "size_kb": file_kb(path),
                    "pos": stats["pos"],
                    "vel": stats["vel"],
                    "grounded_mm": stats["grounded_mm"],
                    "tunnel_n": stats["tunnel_n"],
                    "step_us": stats["neural_us"],
                    "forward_us": stats["forward_us"],
                    "x_analytic": fmt_x_analytic(stats["neural_us"], analytic_us),
                    "pos_delta": pos_delta(fp32_pos, stats["pos"]),
                    "speedup_vs_fp32": fp32_us / stats["neural_us"] if stats["neural_us"] > 0 else 0,
                }
            )
            print(
                f"  {label} {scheme}: pos={stats['pos']:.3f} us={stats['neural_us']:.1f}",
                flush=True,
            )
    return rows


def render_table(rows: list[dict]) -> str:
    lines = [
        PHASE13_START,
        "| Model | Quant | Size KB | Pos err | dpos vs fp32 | Vel err | Tunnel (N) | Step us | x analytic | vs fp32 speed |",
        "|-------|-------|---------|---------|----------------|---------|------------|---------|------------|---------------|",
        "| *(analytic teacher)* | \u2014 | \u2014 | 0 | \u2014 | 0 | 0 | **~0.1** | **1\u00d7** | \u2014 |",
    ]

    order = {s: i for i, s in enumerate(SCHEMES)}
    rows.sort(key=lambda r: (r["label"], order[r["scheme"]]))

    for r in rows:
        speed = f"{r['speedup_vs_fp32']:.2f}\u00d7" if r["scheme"] != "fp32" else "1.00\u00d7"
        mark = " **" if r["scheme"] == "int8_row" and "anchor" in r["label"] else ""
        lines.append(
            f"| {r['label']} | `{r['scheme']}`{mark} | {r['size_kb']:.0f} | {r['pos']:.3f} | "
            f"{r['pos_delta']} | {r['vel']:.3f} | {r['tunnel_n']} | {r['step_us']:.1f} | "
            f"{r['x_analytic']} | {speed} |"
        )

    lines.append(PHASE13_END)
    return "\n".join(lines)


def update_progress_log(table: str) -> None:
    log_path = ROOT / "progress-log.md"
    text = log_path.read_text(encoding="utf-8")
    if PHASE13_START not in text:
        insert = (
            "\n## Phase 13 — Weight quantization sweep\n\n"
            "**Prompt:** Quantize 5 diverse models at several levels; compare speed and rollout fidelity.\n\n"
            "**Schemes:** `fp32` (baseline), `int8_row` (per-row W8A32), `int8_layer` (per-tensor), "
            "`int4_row` (packed W4A32), `fp16` (half weights, FP32 compute).\n\n"
            "Regenerate: `python tools/bench_quant.py`\n\n"
            f"{table}\n"
        )
        marker = "## Expectation timeline"
        text = text.replace(marker, insert + "\n" + marker)
    else:
        start = text.find(PHASE13_START)
        end = text.find(PHASE13_END) + len(PHASE13_END)
        text = text[:start] + table + text[end:]
    log_path.write_text(text, encoding="utf-8")
    print(f"\nUpdated {log_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-quantize", action="store_true", help="Reuse existing quant models")
    args = ap.parse_args()

    if not EXE.exists():
        print(f"Missing {EXE} — build first", file=sys.stderr)
        sys.exit(1)

    if not args.skip_quantize:
        run_quantize()
    rows = build_rows()
    table = render_table(rows)
    print("\n" + table.replace(PHASE13_START, "").replace(PHASE13_END, ""))
    update_progress_log(table)


if __name__ == "__main__":
    main()
