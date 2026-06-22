#!/usr/bin/env python3
"""Benchmark AVX2 SIMD kernels vs pre-SIMD baselines; update Phase 14 in docs/progress-log."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "build" / "voxel_parkour.exe"

PHASE14_START = "<!-- bench-phase14-start -->"
PHASE14_END = "<!-- bench-phase14-end -->"

# Pre-SIMD (Phase 12/13) forward-only us snapshots for comparison.
PRE_SIMD = {
    "3^3 anchor fp32": 12.1,
    "3^3 anchor int8": 14.0,
    "3^3 speed fp32": 2.8,
    "3^3 speed int8": 2.7,
    "9^3 v2 fp32": 79.2,
    "9^3 v2 int8": 88.4,
}

CASES = [
    ("3^3 anchor fp32", "models/patch3/256_rollout8_idle4.bin"),
    ("3^3 anchor int8", "models/quant/int8_row/patch3/256_rollout8_idle4.bin"),
    ("3^3 speed fp32", "models/patch3/64_rollout4_fast.bin"),
    ("3^3 speed int8", "models/quant/int8_row/patch3/64_rollout4_fast.bin"),
    ("9^3 v2 fp32", "models/model_rollout_v2.bin"),
    ("9^3 v2 int8", "models/quant/int8_row/model_rollout_v2.bin"),
]


def bench(path: Path) -> dict:
    text = subprocess.check_output([str(EXE), "--bench", str(path)], cwd=ROOT, text=True)

    def grab(pat):
        m = re.search(pat, text)
        return float(m.group(1)) if m else 0.0

    return {
        "pos": grab(r"Mean position error:\s+([\d.]+)"),
        "forward_us": grab(r"Forward-only \(10k avg\):\s+([\d.]+)"),
        "step_us": grab(r"Avg neural step:\s+([\d.]+)"),
        "kernel": re.search(r"Inference kernel:\s+(\S+)", text).group(1),
    }


def render_table(rows: list[dict]) -> str:
    lines = [
        PHASE14_START,
        "| Config | Pos err | Forward us | Pre-SIMD us | Speedup | Kernel |",
        "|--------|---------|------------|-------------|---------|--------|",
    ]
    for r in rows:
        pre = PRE_SIMD.get(r["label"], 0)
        sp = f"{pre / r['forward_us']:.2f}x" if pre and r["forward_us"] > 0 else "-"
        lines.append(
            f"| {r['label']} | {r['pos']:.3f} | {r['forward_us']:.1f} | {pre:.1f} | {sp} | {r['kernel']} |"
        )
    lines.append(PHASE14_END)
    return "\n".join(lines)


def update_log(table: str) -> None:
    log = ROOT / "docs" / "progress-log.md"
    text = log.read_text(encoding="utf-8")
    if PHASE14_START not in text:
        block = (
            "\n## Phase 14 — AVX2 SIMD inference (final hyper-optimization)\n\n"
            "**Prompt:** Last push — AVX2-FMA FP32 matmul, VNNI int8 experiments, runtime CPU dispatch.\n\n"
            "**Implementation:** `neural_simd.c` — AVX2 8-wide FP32 dots, fused layer-1 norm scratch, "
            "runtime `__builtin_cpu_supports` dispatch. VNNI W8A8 tested and rejected (rollout blow-up); "
            "int8 kept on scalar W8A32 path.\n\n"
            "Regenerate: `python tools/bench_simd.py`\n\n"
            f"{table}\n"
        )
        text = text.replace("## Expectation timeline", block + "\n## Expectation timeline")
    else:
        s = text.find(PHASE14_START)
        e = text.find(PHASE14_END) + len(PHASE14_END)
        text = text[:s] + table + text[e:]
    log.write_text(text, encoding="utf-8")
    print(f"Updated {log}", flush=True)


def main():
    if not EXE.exists():
        print(f"Missing {EXE}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for label, rel in CASES:
        stats = bench(ROOT / rel)
        rows.append({"label": label, **stats})
        print(f"  {label}: pos={stats['pos']:.3f} fwd={stats['forward_us']:.1f}us", flush=True)

    table = render_table(rows)
    print("\n" + table.replace(PHASE14_START, "").replace(PHASE14_END, ""))
    update_log(table)


if __name__ == "__main__":
    main()
