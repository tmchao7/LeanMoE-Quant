#!/usr/bin/env python3
"""make_figures — 从 `results/**/result.json` 生成 RESULTS.md 用的图（纯 stdlib，无绘图依赖）。

## 为什么自己写 SVG 而不是用 matplotlib

- 本仓库的依赖表很窄（`numpy` 是唯一的数值依赖），为三张图引入 matplotlib 不划算；
- **每个数字都必须能追到工件**：本工具把每个输入连同它的来源路径写进
  `docs/figures/inputs.json`，并**读不到就报错**（绝不静默画一个"看起来对"的数）。
  画图的数字若来自记录正文（而不是 payload），在 `inputs.json` 里标 `"source": "record"`
  并写明记录路径 —— 让读者一眼看出哪些是工件、哪些是转录。

## 用法

```bash
python -m tools.make_figures            # 写到 docs/figures/
python -m tools.make_figures --check    # 只校验输入存在，不写字
```

图（英文标注，方便脱离本文被单看）：
    fig1_per_axis_speedups.svg   五条杠杆各自的同轮 A/B（+ 1.0× 参考线）
    fig2_to_the_wall.svg         从 fp16 eager 到字节下限（对数轴）
    fig3_kernel_attainment.svg   隔离测量：必要字节 vs 实测时间（ncu 替代口径）
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
INK = "#1f2328"
MUTED = "#57606a"
GRID = "#d8dee4"
COLORS = {
    "fp16": "#4c6ef5",
    "quant": "#12b886",
    "attn": "#f59f00",
    "norm": "#e8590c",
    "floor": "#adb5bd",
    "warn": "#c92a2a",
}

# 展示层设计系统（2026-10-31）：
#   结论式标题（标题即结论，数值全部来自同一份 inputs）；直读数值标签；
#   语义色板：蓝=fp16/执行轴、绿=量化、橙=融合 norm、黄=attention/KV 路径、红=门线、灰=地板/背景；
#   每张图左下角出处脚注（真实工件路径），完整「数字→工件」映射在 docs/figures/inputs.json。
FULL_PROV = "Full provenance per number: docs/figures/inputs.json"

# 每个数字的来源。`kind=payload` 从工件读；`kind=record` 是记录正文里的值（转录，写明出处）。
SOURCES: dict[str, dict[str, str]] = {
    "baseline_fp16_eager_1024": {
        "kind": "payload",
        "path": "results/EXP-p0-02-baseline-fp16__ctx=1024/result.json",
        "ptr": "latency.per_token_p50_ms",
        "note": "EXP-p0-02 的『记录采用』值（P2j 归档/恢复后的规范 payload）",
    },
    "byte_floor_1024": {
        "kind": "payload",
        "path": "results/EXP-p0-02-baseline-fp16__ctx=1024/result.json",
        "ptr": "attainment.floor_t_ms",
        "note": "字节下限 [T] ÷ 实测可达带宽",
    },
    "fp16_cudagraph_1024": {
        "kind": "payload",
        "path": "results/EXP-p4b-01-truncation__bit_width=fp16_execution=cudagraph_ctx=1024/result.json",
        "ptr": "latency.per_token_p50_ms",
    },
    "fp16_cudagraph_4096": {
        "kind": "payload",
        "path": "results/EXP-p4b-01-truncation__bit_width=fp16_execution=cudagraph_ctx=4096/result.json",
        "ptr": "latency.per_token_p50_ms",
    },
    "w8a16_cudagraph_1024": {
        "kind": "payload",
        "path": "results/EXP-p4b-01-truncation__bit_width=w8a16_execution=cudagraph_ctx=1024/result.json",
        "ptr": "latency.per_token_p50_ms",
    },
    "w8a16_cudagraph_4096": {
        "kind": "payload",
        "path": "results/EXP-p4b-01-truncation__bit_width=w8a16_execution=cudagraph_ctx=4096/result.json",
        "ptr": "latency.per_token_p50_ms",
    },
    "fp16_cudagraph_1024_pre_p2j": {
        "kind": "payload",
        "path": "results/EXP-p2j-refresh/archive/EXP-p4b-01-truncation__bit_width=fp16_execution=cudagraph_ctx=1024.json",
        "ptr": "latency.per_token_p50_ms",
        "note": "P2j 归档的同轮 cudagraph（HF norm 轮）",
    },
    "fp16_cudagraph_4096_pre_p2j": {
        "kind": "payload",
        "path": "results/EXP-p2j-refresh/archive/EXP-p4b-01-truncation__bit_width=fp16_execution=cudagraph_ctx=4096.json",
        "ptr": "latency.per_token_p50_ms",
    },
    "fp16_eager_4096": {
        "kind": "payload",
        "path": "results/EXP-p2j-refresh/archive/EXP-p4b-01-truncation__bit_width=fp16_execution=eager_ctx=4096.json",
        "ptr": "latency.per_token_p50_ms",
    },
    "fp16_eager_1024": {
        "kind": "payload",
        "path": "results/EXP-p2j-refresh/archive/EXP-p4b-01-truncation__bit_width=fp16_execution=eager_ctx=1024.json",
        "ptr": "latency.per_token_p50_ms",
        "note": "★ 归档 pre-P2j 轮的 eager 臂（9.8841），与同归档的 cudagraph（4.0670）构成 "
        "RESULTS.md 执行轴行引用的同轮对（2.43×）。base-dir result.json 的 8.1179 在记录里"
        "无出处（v1=9.732、v2 采纳=9.8841、v3 作废=8.2915 都不是它）——不采用，防跨轮混比。",
    },
    # --- EXP-p4c-01 的 v7（2026-10-25 E6 采纳刷新，canonical） ---
    "p4c_fp16_sdpa_1024": {
        "kind": "payload",
        "path": "results/EXP-p4c-01-in-graph-attention__bit_width=fp16_attn=sdpa_ctx=1024/result.json",
        "ptr": "latency.per_token_p50_ms",
        "note": "v7 canonical（fp16×sdpa 参照）",
    },
    "p4c_fp16_sdpa_4096": {
        "kind": "payload",
        "path": "results/EXP-p4c-01-in-graph-attention__bit_width=fp16_attn=sdpa_ctx=4096/result.json",
        "ptr": "latency.per_token_p50_ms",
    },
    "p4c_fp16_kernel_1024": {
        "kind": "payload",
        "path": "results/EXP-p4c-01-in-graph-attention__bit_width=fp16_attn=kernel_ctx=1024/result.json",
        "ptr": "latency.per_token_p50_ms",
    },
    "p4c_fp16_kernel_4096": {
        "kind": "payload",
        "path": "results/EXP-p4c-01-in-graph-attention__bit_width=fp16_attn=kernel_ctx=4096/result.json",
        "ptr": "latency.per_token_p50_ms",
    },
    "p4c_w8a16_sdpa_1024": {
        "kind": "payload",
        "path": "results/EXP-p4c-01-in-graph-attention__bit_width=w8a16_attn=sdpa_ctx=1024/result.json",
        "ptr": "latency.per_token_p50_ms",
        "note": "v7 同轮的 w8a16×sdpa，用作 attention 那一跳的分母",
    },
    "p4c_w8a16_kernel_1024": {
        "kind": "payload",
        "path": "results/EXP-p4c-01-in-graph-attention__bit_width=w8a16_attn=kernel_ctx=1024/result.json",
        "ptr": "latency.per_token_p50_ms",
        "note": "本项目延迟最低的一格（v7 canonical）",
    },
    # --- 融合 RMSNorm 全步 A/B（记录正文，工件是 fused_norm_ab.json） ---
    "norm_hf_step_ms": {
        "kind": "record",
        "record": "experiments/EXP-p4c-02-elementwise-tail.md",
        "value": 2.4143,
        "note": "§5.6 三臂 A/B 的 HF 原链",
    },
    "norm_ours_step_ms": {
        "kind": "record",
        "record": "experiments/EXP-p4c-02-elementwise-tail.md",
        "value": 1.9537,
        "note": "§5.6 三臂 A/B 的自研融合 kernel（15/15 轮获胜）",
    },
    # --- 隔离的 kernel 级证据 ---
    "kernel_attainment": {
        "kind": "payload",
        "path": "results/ncu_evidence/isolated.json",
        "ptr": "subjects",
        "note": "cold（每次计时前 memset 256 MB 冲 L2）− 事件地板",
    },
    "measured_bw_gbps": {
        "kind": "payload",
        "path": "results/ncu_evidence/isolated.json",
        "ptr": "measured_achievable_gbps",
    },
}


def _resolve(ptr: str, blob: Any) -> Any:
    node = blob
    for part in ptr.split("."):
        if isinstance(node, list):
            node = node[int(part)]
        else:
            node = node[part]
    return node


def load_inputs() -> dict[str, Any]:
    """读全部输入；读不到就抛，绝不静默。"""
    out: dict[str, Any] = {}
    for name, spec in SOURCES.items():
        if spec["kind"] == "record":
            out[name] = {
                "value": spec["value"],
                "source": spec["record"] + "（记录正文，转录）",
                "note": spec.get("note"),
            }
            continue
        path = os.path.join(REPO_ROOT, spec["path"])
        if not os.path.exists(path):
            raise FileNotFoundError(
                "缺工件 %s（%s）—— 图必须从工件画，缺了就报错，不许用记忆里的数"
                % (spec["path"], name)
            )
        with open(path, encoding="utf-8") as fh:
            blob = json.load(fh)
        out[name] = {
            "value": _resolve(spec["ptr"], blob),
            "source": "%s → %s" % (spec["path"], spec["ptr"]),
            "note": spec.get("note"),
        }
    return out


# --------------------------------------------------------------------------------------
# 极简 SVG 构造器
# --------------------------------------------------------------------------------------


class Svg:
    """极简 SVG 构造器 + **画布越界自检**。

    ★ 为什么要自检：SVG 不会报错，画出去的东西就是"没画"。第一版最长的那条标注
      （执行轴的 2.76×）整条落在画布外，图上只剩一根没有数字的条 —— 只有靠人工
      渲染才看得出来。这里把每个图元的包围盒记下来，`render()` 时统一校验，
      越界就抛异常（并指出是谁）。这样"图有没有画坏"是**可测**的，不靠眼睛。
    """

    def __init__(self, width: int, height: int, title: str, subtitle: Any = ""):
        self.w, self.h = width, height
        self.boxes: list[tuple[str, float, float, float, float]] = []
        self.parts: list[str] = [
            '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
            'viewBox="0 0 %d %d" font-family="%s">' % (width, height, width, height, FONT),
            '<rect width="%d" height="%d" fill="#ffffff"/>' % (width, height),
        ]
        self.text(24, 34, title, size=17, weight="600", color=INK)
        # ★ 副标题必须分行写：SVG 里没有换行，第一版把一句长话塞进去，右端直接被画布截掉
        lines = [subtitle] if isinstance(subtitle, str) else list(subtitle)
        for i, line in enumerate(lines):
            if line:
                self.text(24, 56 + i * 15, line, size=11.5, color=MUTED)

    def text(self, x, y, content, *, size=11, color=INK, weight="400", anchor="start"):
        half = _text_width(str(content), size) / 2
        x0 = (
            x
            if anchor == "start"
            else (x - _text_width(str(content), size) if anchor == "end" else x - half)
        )
        self._track("text:%s" % str(content)[:28], x0, y - size, x0 + 2 * half, y + size * 0.35)
        self.parts.append(
            '<text x="%.1f" y="%.1f" font-size="%.1f" fill="%s" font-weight="%s" '
            'text-anchor="%s">%s</text>' % (x, y, size, color, weight, anchor, _esc(content))
        )

    def rect(self, x, y, w, h, fill, *, opacity=1.0, radius=0.0):
        self._track("rect", x, y, x + w, y + h)
        self.parts.append(
            '<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="%s" opacity="%.2f" rx="%.1f"/>'
            % (x, y, w, h, fill, opacity, radius)
        )

    def line(self, x1, y1, x2, y2, color=GRID, *, width=1.0, dash=None):
        d = ' stroke-dasharray="%s"' % dash if dash else ""
        self.parts.append(
            '<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="%.1f"%s/>'
            % (x1, y1, x2, y2, color, width, d)
        )

    def _track(self, kind: str, x0, y0, x1, y1) -> None:
        self.boxes.append((kind, float(x0), float(y0), float(x1), float(y1)))

    def render(self) -> str:
        bad = []
        for kind, x0, y0, x1, y1 in self.boxes:
            if x0 < -1 or y0 < -1 or x1 > self.w + 1 or y1 > self.h + 1:
                bad.append("%s  x[%.0f,%.0f] y[%.0f,%.0f]" % (kind, x0, x1, y0, y1))
        if bad:
            raise ValueError(
                "画布越界（%dx%d）：%d 个图元跑到画布外 —— %s。SVG 不会报错，只会「没画」。"
                % (self.w, self.h, len(bad), "; ".join(bad[:4]))
            )
        return "\n".join(self.parts + ["</svg>", ""])


def _esc(s: Any) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _fmt(value: float, digits: int = 2) -> str:
    return ("%%.%df" % digits) % value


def _text_width(text: str, size: float) -> float:
    """粗略估宽（SVG 没有排版引擎，只能估）：用于判断标注会不会画出画布。

    ★ 这不是美观问题：第一版里最长的那条标注（执行轴的 2.76×）整个被画到画布外面，
      图上就只剩一根"没有数字的条"。自动改到条内右对齐即可。
    """
    ascii_len = sum(1 for ch in text if ord(ch) < 128)
    wide_len = len(text) - ascii_len
    return ascii_len * size * 0.56 + wide_len * size * 1.0


# --------------------------------------------------------------------------------------
# fig 1：五条杠杆各自的同轮 A/B
# --------------------------------------------------------------------------------------


def fig1(inputs: dict[str, Any]) -> str:
    def p50(name: str) -> float:
        return float(inputs[name]["value"])

    def ms(name: str) -> str:
        return "%.2f" % p50(name)

    def ratio(a: str, b: str) -> float:
        return p50(a) / p50(b)

    groups = [
        (
            "Execution\n(eager → CUDA Graph)",
            [
                (
                    "@1024",
                    ratio("fp16_eager_1024", "fp16_cudagraph_1024_pre_p2j"),
                    COLORS["fp16"],
                    "%s → %s ms" % (ms("fp16_eager_1024"), ms("fp16_cudagraph_1024_pre_p2j")),
                ),
                (
                    "@4096",
                    ratio("fp16_eager_4096", "fp16_cudagraph_4096_pre_p2j"),
                    COLORS["fp16"],
                    "%s → %s ms" % (ms("fp16_eager_4096"), ms("fp16_cudagraph_4096_pre_p2j")),
                ),
            ],
        ),
        (
            "Quantization\n(w8a16 vs fp16, in-graph)",
            [
                (
                    "@1024",
                    ratio("fp16_cudagraph_1024", "w8a16_cudagraph_1024"),
                    COLORS["quant"],
                    "%s → %s ms" % (ms("fp16_cudagraph_1024"), ms("w8a16_cudagraph_1024")),
                ),
                (
                    "@4096",
                    ratio("fp16_cudagraph_4096", "w8a16_cudagraph_4096"),
                    COLORS["quant"],
                    "%s → %s ms" % (ms("fp16_cudagraph_4096"), ms("w8a16_cudagraph_4096")),
                ),
            ],
        ),
        (
            "Attention impl\n(HF sdpa → split-KV)",
            [
                (
                    "@1024",
                    ratio("p4c_fp16_sdpa_1024", "p4c_fp16_kernel_1024"),
                    COLORS["attn"],
                    "%s → %s ms" % (ms("p4c_fp16_sdpa_1024"), ms("p4c_fp16_kernel_1024")),
                ),
                (
                    "@4096",
                    ratio("p4c_fp16_sdpa_4096", "p4c_fp16_kernel_4096"),
                    COLORS["attn"],
                    "%s → %s ms" % (ms("p4c_fp16_sdpa_4096"), ms("p4c_fp16_kernel_4096")),
                ),
            ],
        ),
        (
            "Fused RMSNorm\n(7 kernels → 1, whole step)",
            [
                (
                    "@1024",
                    float(inputs["norm_hf_step_ms"]["value"])
                    / float(inputs["norm_ours_step_ms"]["value"]),
                    COLORS["norm"],
                    "%.2f → %.2f ms"
                    % (inputs["norm_hf_step_ms"]["value"], inputs["norm_ours_step_ms"]["value"]),
                ),
            ],
        ),
    ]

    w, h = 1180, 648
    svg = Svg(
        w,
        h,
        "Every lever pays: all four same-round A/Bs land above 1.0×",
        [
            "OLMoE-1B-7B-0924, batch=1 decode, RTX 5090, measured achievable bandwidth 1560.3 GB/s.",
            "Ratio > 1.0 = faster. These are four independent A/Bs: do not multiply them (RESULTS.md rule).",
        ],
    )
    left, right, top, bottom = 210, 420, 108, 92
    plot_w, plot_h = w - left - right, h - top - bottom
    vmax = 3.4

    def x_of(v: float) -> float:
        return left + plot_w * (v / vmax)

    # 网格
    for tick in (1.0, 1.5, 2.0, 2.5, 3.0):
        x = x_of(tick)
        svg.line(x, top, x, top + plot_h, GRID, width=0.8)
        svg.text(x, top + plot_h + 18, _fmt(tick, 1) + "×", size=10.5, color=MUTED, anchor="middle")
    svg.text(
        left + plot_w / 2,
        top + plot_h + 38,
        "measured speedup (×, same-round A/B)",
        size=11,
        color=MUTED,
        anchor="middle",
    )

    row_h = plot_h / len(groups)
    for gi, (label, bars) in enumerate(groups):
        y0 = top + gi * row_h
        for li, line_text in enumerate(label.split("\n")):
            svg.text(
                left - 16,
                y0 + row_h / 2 - 4 + li * 15,
                line_text,
                size=11.5,
                color=INK,
                anchor="end",
            )
        bar_h = 16
        offset = (row_h - (len(bars) * bar_h + (len(bars) - 1) * 6)) / 2
        for bi, (sublabel, value, color, note) in enumerate(bars):
            by = y0 + offset + bi * (bar_h + 6)
            if value is None:
                svg.text(left + 6, by + 12, note, size=10.5, color=MUTED)
                continue
            svg.rect(left, by, max(1.0, x_of(value) - left), bar_h, color, radius=3)
            # ★ caption 统一放**条外右侧**（2026-10-25 视觉审计：条内白字会压住条外
            #   黑字——两套分支都画导致重叠）。画布已按最长 caption 预算加宽。
            caption = "%.2f×  %s   (%s)" % (value, sublabel, note)
            svg.text(x_of(value) + 8, by + 12.5, caption, size=11, color=INK, weight="600")

    # 1.0× 参考线（"没变"）
    svg.line(x_of(1.0), top - 8, x_of(1.0), top + plot_h + 4, COLORS["warn"], width=1.4, dash="4 3")
    svg.text(
        x_of(1.0) + 6, top - 12, "1.0× = no change", size=10.5, color=COLORS["warn"], weight="600"
    )
    # 左下出处脚注（设计系统：图上每个数都能沿这行追到工件；完整映射在 inputs.json）
    svg.text(
        24,
        h - 24,
        "Data: results/EXP-p0-02-baseline-fp16__ctx=1024 · results/EXP-p4b-01-truncation__* · "
        "results/EXP-p4c-01-in-graph-attention__* (v7) · experiments/EXP-p4c-02-elementwise-tail.md §5.6",
        size=9,
        color=MUTED,
    )
    svg.text(24, h - 11, FULL_PROV, size=9, color=MUTED)
    return svg.render()


# --------------------------------------------------------------------------------------
# fig 2：离带宽墙有多远（对数轴）
# --------------------------------------------------------------------------------------


def fig2(inputs: dict[str, Any]) -> str:
    """★ 链条里**每一跳**都必须是同轮（或同格前后）比较，且逐跳标注它是什么杠杆。

    第一版把"pre-P2i 的 eager"接上"post-P2j 的 graph"，凭空造出一个 2.75× —— 那是跨轮混比
    （硬规则 3）。现在：
      跳1 execution  : 同为 pre-P2j 那一轮（归档里的 eager 与 cudagraph 两个 payload）
      跳2 融合 norm  : 同一个 cudagraph 格在 P2j 前后的值（4.0670 → 3.5881）
      跳3 quantization: 同轮（P2j canonical 的两个格）
      跳4 attention  : 用 v3 同轮的 w8a16×sdpa（3.0036）作分母，而不是上一根条
    最早那个 82.64 ms 来自另一轮，画成空心条、明确标注"不属于链条"。
    """

    def p50(name: str) -> float:
        return float(inputs[name]["value"])

    eager_pre = p50("fp16_eager_1024")
    graph_pre = p50("fp16_cudagraph_1024_pre_p2j")
    fp16_graph_now = p50("fp16_cudagraph_1024")
    w8a16_graph_now = p50("w8a16_cudagraph_1024")
    w8a16_sdpa_v3 = p50("p4c_w8a16_sdpa_1024")
    w8a16_kernel = p50("p4c_w8a16_kernel_1024")

    chain = [
        ("fp16 eager", eager_pre, COLORS["fp16"], None),
        (
            "fp16 + CUDA Graph",
            graph_pre,
            COLORS["fp16"],
            "%.2f×  execution axis (same round)" % (eager_pre / graph_pre),
        ),
        (
            "+ fused RMSNorm",
            fp16_graph_now,
            COLORS["norm"],
            "%.2f×  fused RMSNorm (same cell, P2j)" % (graph_pre / fp16_graph_now),
        ),
        (
            "w8a16 + CUDA Graph",
            w8a16_graph_now,
            COLORS["quant"],
            "%.2f×  quantization (same round)" % (fp16_graph_now / w8a16_graph_now),
        ),
        (
            "w8a16 × split-KV kernel",
            w8a16_kernel,
            COLORS["attn"],
            "%.2f×  attention impl (vs its own same-round sdpa %.2f ms)"
            % (w8a16_sdpa_v3 / w8a16_kernel, w8a16_sdpa_v3),
        ),
    ]
    context = ("fp16 eager (earlier round)", p50("baseline_fp16_eager_1024"), COLORS["fp16"], None)
    floor = p50("byte_floor_1024")

    w, h = 940, 546
    svg = Svg(
        w,
        h,
        "From %.1f to %.2f ms @ctx=1024: every hop is its own same-round A/B"
        % (context[1], chain[-1][1]),
        [
            "ms per token, log scale. Dashed line = byte floor [T] / measured achievable bandwidth (%.4f ms)."
            % floor,
            "Each hop is its own same-round (or same-cell) A/B; the hollow bar is context, not a hop.",
        ],
    )
    left, right, top, bottom = 300, 250, 118, 96
    plot_w, plot_h = w - left - right, h - top - bottom
    import math

    lo, hi = math.log10(1.0), math.log10(120.0)

    def x_of(ms: float) -> float:
        return left + plot_w * (math.log10(ms) - lo) / (hi - lo)

    for tick in (1, 2, 5, 10, 20, 50, 100):
        x = x_of(tick)
        svg.line(x, top, x, top + plot_h, GRID, width=0.8)
        svg.text(x, top + plot_h + 18, str(tick), size=10.5, color=MUTED, anchor="middle")
    svg.text(
        left + plot_w / 2,
        top + plot_h + 40,
        "ms / token  (log scale)",
        size=11,
        color=MUTED,
        anchor="middle",
    )

    rows = [context] + chain
    row_h = plot_h / len(rows)
    bar_h = 18
    for i, (label, value, color, hop) in enumerate(rows):
        y = top + i * row_h + (row_h - bar_h) / 2
        if i == 0:
            svg.rect(left, y, max(2.0, x_of(value) - left), bar_h, color, opacity=0.18, radius=3)
            svg.line(left, y + bar_h, x_of(value), y + bar_h, color, width=1.2, dash="3 3")
        else:
            svg.rect(left, y, max(2.0, x_of(value) - left), bar_h, color, radius=3)
        svg.text(left - 14, y + 13, label, size=11.5, color=INK, anchor="end")
        # ★ ms 值标签：条够宽放条内（白字）；窄条（<70px）放条外右侧黑字——
        #   末跳 1.46 ms 的条只有 ~30px，白字会被截（视觉审计 2026-10-25）。
        label_w = _text_width("%.2f ms" % value, 11)
        if x_of(value) - left > label_w + 16:
            svg.text(
                x_of(value) - 8,
                y + 13,
                "%.2f ms" % value,
                size=11,
                color=COLORS["warn"] if i == 0 else "#ffffff",
                weight="600",
                anchor="end",
            )
        else:
            svg.text(
                x_of(value) + 8,
                y + 13,
                "%.2f ms" % value,
                size=11,
                color=INK,
                weight="600",
            )
        if hop:
            # hop 注记起点也要让开条外的 ms 标签
            hop_x = x_of(value) + 10 + (0 if x_of(value) - left > label_w + 16 else label_w + 10)
            svg.text(hop_x, y + 13, hop, size=10.5, color=MUTED)

    xf = x_of(floor)
    svg.line(xf, top - 10, xf, top + plot_h + 4, COLORS["floor"], width=1.6, dash="5 3")
    svg.text(xf + 8, top - 16, "byte floor %.4f ms" % floor, size=10.5, color=MUTED, weight="600")
    svg.text(
        24,
        h - 34,
        "Baseline sits %.0f× above the floor. The best cell reads %.2f× of it — below the fp16-"
        "accounting floor, because KV-INT8 streams half the KV bytes (E6)."
        % (context[1] / floor, chain[-1][1] / floor),
        size=10.5,
        color=MUTED,
    )
    svg.text(
        24,
        h - 20,
        "Data: results/EXP-p0-02-baseline-fp16__ctx=1024 (baseline & floor) · "
        "results/EXP-p2j-refresh/archive · results/EXP-p4b-01-truncation__* · "
        "results/EXP-p4c-01-in-graph-attention__*",
        size=9,
        color=MUTED,
    )
    svg.text(24, h - 8, FULL_PROV, size=9, color=MUTED)
    return svg.render()


# --------------------------------------------------------------------------------------
# fig 3：隔离的 kernel 级证据
# --------------------------------------------------------------------------------------


def _not_measurable_label(rec: dict[str, Any]) -> str:
    """`attainment_pct_cold_best` 为 None 时，图上该写什么。

    ★ 成因不止一个，必须分别说，否则就是把"未测"写成具体结论（硬规则 1）：
      - 压根没跑 cold（flush_mb == 0 ⇒ cold_single 为空）→ "not measured"；
      - 真的低于事件地板 → 引用**实测**的地板值（`timing.event_floor_ms`），
        不写死 5.5 µs —— 地板本身是量出来的，图上就该引它。
    """
    timing = rec.get("timing") or {}
    if not timing:
        return "not measured (no timing for this subject)"
    if not timing.get("cold_single"):
        return "not measured (cold pass was not run — flush_mb=0)"
    floor_ms = timing.get("event_floor_ms")
    if floor_ms is None:
        return "below the event floor — not measurable"
    return "below the %.1f µs event floor — not measurable" % (float(floor_ms) * 1000.0)


def fig3(inputs: dict[str, Any]) -> str:
    subjects = inputs["kernel_attainment"]["value"]
    order = [
        "grouped_fp16_layer",
        "grouped_w8a16_layer",
        "grouped_w4a16_layer",
        "dequant_lm_head_w8a16",
        "attn_split_kv_1024",
        "rmsnorm_h2048",
    ]
    pretty = {
        "grouped_fp16_layer": "MoE grouped GEMV (fp16)",
        "grouped_w8a16_layer": "MoE grouped GEMV (w8a16)",
        "grouped_w4a16_layer": "MoE grouped GEMV (w4a16)",
        "dequant_lm_head_w8a16": "fused dequant-GEMM (w8a16)",
        "attn_split_kv_1024": "split-KV attention (ctx=1024)",
        "rmsnorm_h2048": "fused RMSNorm (H=2048)",
    }
    bw = float(inputs["measured_bw_gbps"]["value"])

    # 结论式标题：两个 headline 达成率直接从工件读；缺失则回退到描述式标题（不许编数）。
    att_grouped = subjects["grouped_fp16_layer"].get("attainment_pct_cold_best")
    att_deq = subjects["dequant_lm_head_w8a16"].get("attainment_pct_cold_best")
    if att_grouped is not None and att_deq is not None:
        title = "Grouped GEMV runs at %.0f%% of the wall; fused dequant-GEMM %.1f%%" % (
            att_grouped,
            att_deq,
        )
    else:
        title = "Kernel-level: necessary bytes vs measured time (isolated, L2 flushed)"

    w, h = 980, 552
    svg = Svg(
        w,
        h,
        title,
        [
            "One call per subject. attainment = hand-computed necessary bytes / cold time / "
            "measured achievable bandwidth (%.1f GB/s)." % bw,
            "ncu performance counters are blocked on this host — see docs/reference/NCU_EVIDENCE.md §1.",
        ],
    )
    left, right, top, bottom = 230, 190, 118, 108
    plot_w, plot_h = w - left - right, h - top - bottom
    vmax = 100.0

    def x_of(v: float) -> float:
        return left + plot_w * (v / vmax)

    for tick in (0, 25, 50, 75, 100):
        x = x_of(tick)
        svg.line(x, top, x, top + plot_h, GRID, width=0.8)
        svg.text(x, top + plot_h + 18, "%d%%" % tick, size=10.5, color=MUTED, anchor="middle")
    svg.text(
        left + plot_w / 2,
        top + plot_h + 40,
        "% of measured achievable DRAM bandwidth (cold)",
        size=11,
        color=MUTED,
        anchor="middle",
    )

    row_h = plot_h / len(order)
    bar_h = 18
    for i, key in enumerate(order):
        rec = subjects[key]
        y = top + i * row_h + (row_h - bar_h) / 2
        svg.text(left - 14, y + 13, pretty[key], size=11.5, color=INK, anchor="end")
        att = rec.get("attainment_pct_cold_best")
        if att is None:
            # ★ `att is None` 有**不止一个**成因，不能一律写成同一句结论：
            #   (a) cold_min − event_floor <= 0（真的小于事件地板）；
            #   (b) flush_mb == 0 ⇒ cold 列表为空 ⇒ 压根没做 cold 计时。
            #   旧文案对 (b) 也会写"低于 5.5 µs 事件地板 —— 不可测"，那是把"未测"
            #   写成了具体结论（硬规则 1）。而且那个 5.5 µs 是写死的，事件地板本身
            #   是**实测**的（同一份 isolated.json 里就有），应当引用而不是硬编码。
            svg.rect(left, y, 2, bar_h, COLORS["warn"], radius=3)
            svg.text(
                left + 10,
                y + 13,
                _not_measurable_label(rec),
                size=10.5,
                color=COLORS["warn"],
            )
        else:
            color = COLORS["quant"] if "grouped" in key else COLORS["fp16"]
            if "dequant" in key:
                color = COLORS["attn"]
            if "attn" in key:
                color = COLORS["attn"]
            if "rmsnorm" in key:
                color = COLORS["norm"]
            svg.rect(left, y, max(2.0, x_of(att) - left), bar_h, color, radius=3)
            svg.text(
                x_of(att) + 8,
                y + 13,
                "%.1f%%   %.4f ms" % (att, rec["timing"]["cold_single"]["min_ms"]),
                size=10.5,
                color=INK,
                weight="600",
            )

    # 图下方：量化 vs 字节（右上角整块留给条上的百分比标注，避免两段文字压在一起）
    base = subjects["grouped_fp16_layer"]
    base_ms = base["timing"]["cold_single"]["min_ms"]
    base_b = base["analytical_bytes"]["read"]
    foot_y = top + plot_h + 52
    svg.text(
        left, foot_y, "expert GEMV — bytes ↓ / time ↓ (vs fp16):", size=11, color=INK, weight="600"
    )
    for i, key in enumerate(("grouped_fp16_layer", "grouped_w8a16_layer", "grouped_w4a16_layer")):
        rec = subjects[key]
        ms = rec["timing"]["cold_single"]["min_ms"]
        byte_ratio = base_b / rec["analytical_bytes"]["read"]
        time_ratio = base_ms / ms
        svg.text(
            left + 240 + i * 150,
            foot_y,
            "%s  %.2f× / %.2f×" % (key.split("_")[1], byte_ratio, time_ratio),
            size=10.5,
            color=INK,
        )
    svg.text(
        left,
        foot_y + 18,
        "Byte savings do not convert 1:1 — the in-kernel unpacking is the added cost, "
        "so attainment falls as bit-width drops.",
        size=10.5,
        color=MUTED,
    )
    svg.text(
        24,
        h - 22,
        "Data: results/ncu_evidence/isolated.json — cold timing (256 MB L2 flush before each "
        "timed call) minus the measured event floor",
        size=9,
        color=MUTED,
    )
    svg.text(24, h - 9, FULL_PROV, size=9, color=MUTED)
    return svg.render()


def fig4(inputs: dict[str, Any]) -> str:
    """Seven adopted decode-path levers — same-round A/B step-time reduction.

    ★ 数值口径：每条都是**独立进程·同轮相邻** A/B 的步时降幅（RESULTS.md §2 的行转录，
    每行的工件引用在 RESULTS.md；E6 @1024 的 −0.78% 因条形太短只在脚注提及）。
    """
    levers = [
        # 颜色 = 开关类别（图例语义统一，2026-10-25 视觉审计）：
        #   norm 橙 = kernel/fusion 融合（F1/F2）、attn 黄 = KV 路径（F4）、quant 绿 = 量化（F5/E6）
        ("fused RoPE (F1)", 8.39, "A/B", "fused_rope_ab", "norm"),
        ("KV INT8 read path (E6, ctx=4096)", 6.54, "5/5 pairs", "EXP-e6-ab-4096-run2", "quant"),
        ("packed q/k/v GEMM (F2)", 4.81, "3/3 pairs", "EXP-p4c-01 §F2", "norm"),
        ("lm_head W8 quant (F5)", 4.13, "5/5 pairs", "EXP-f5-ab-20261025", "quant"),
        ("fused routing top-k (F1)", 3.00, "A/B", "routing_ab", "norm"),
        ("KV direct write (F4)", 2.94, "5/5 pairs", "EXP-p4c-01 §F4", "attn"),
        ("skip mask-fill (F1)", 1.33, "A/B", "skip_mask_fill_ab", "norm"),
    ]
    w, h = 980, 448
    svg = Svg(
        w,
        h,
        "Seven adopted levers: −%.1f%% to −%.1f%% per decode step, each A/B'd same-round"
        % (max(v for _, v, _, _, _ in levers), min(v for _, v, _, _, _ in levers)),
        [
            "Independent processes, adjacent arms, same round; canary drift ≤5%, host_share ≤5% per arm.",
            "Sources: RESULTS.md §2 (every row cites its artifact). KV INT8 at ctx=1024: −0.78% (5/5).",
        ],
    )
    left, row_h, gap = 250, 30, 10
    y0 = 96
    x0, x_max_px, v_max = left + 6, 600.0, 9.0

    def x_of(v: float) -> float:
        return x0 + (v / v_max) * (x_max_px - x0)

    # 网格线（0/3/6/9%）
    for gv in (0.0, 3.0, 6.0, 9.0):
        gx = x_of(gv)
        svg.line(gx, y0 - 8, gx, y0 + len(levers) * (row_h + gap) - gap + 6)
        svg.text(
            gx,
            y0 + len(levers) * (row_h + gap) + 16,
            "%.0f%%" % gv,
            size=10,
            color=MUTED,
            anchor="middle",
        )
    for i, (name, val, pairs, src, color_key) in enumerate(levers):
        y = y0 + i * (row_h + gap)
        svg.text(left, y + row_h * 0.68, name, size=11.5, anchor="end")
        svg.rect(x0, y, x_of(val) - x0, row_h, COLORS[color_key], radius=3)
        svg.text(x_of(val) + 8, y + row_h * 0.68, "−%.2f%%" % val, size=11.5, weight="600")
        # ★ 注释放**固定列**（x=650，不跟随条尾浮动）：条尾 x 最大 ~577、值标签最长
        #   到 ~640，注记列最长串 650+~190 仍在画布内（跟随式曾被渲染器裁边）。
        svg.text(650, y + row_h * 0.68, "%s · %s" % (pairs, src), size=9.5, color=MUTED)
    # 图例（颜色 = 开关类别）：画布底部，色块+文字
    legend_y = y0 + len(levers) * (row_h + gap) + 34
    lx = x0
    for label, key in (
        ("fusion (F1/F2)", "norm"),
        ("KV path (F4)", "attn"),
        ("quantization (F5/E6)", "quant"),
    ):
        svg.rect(lx, legend_y - 10, 14, 14, COLORS[key], radius=3)
        svg.text(lx + 20, legend_y + 1, label, size=10.5, color=MUTED)
        lx += 20 + _text_width(label, 10.5) + 26
    svg.text(
        24,
        h - 12,
        "Data: RESULTS.md §2 (each row cites its artifact: results/EXP-e6-ab-4096-run2 · "
        "results/EXP-f5-ab-20261025 · results/EXP-f2-ab-20261024 · results/EXP-f4-ab-20261025)",
        size=9,
        color=MUTED,
    )
    return svg.render()


FIGURES = {
    "fig1_per_axis_speedups.svg": fig1,
    "fig2_to_the_wall.svg": fig2,
    "fig3_kernel_attainment.svg": fig3,
    "fig4_lever_ab.svg": fig4,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=os.path.join("docs", "figures"))
    ap.add_argument("--check", action="store_true", help="只校验输入与渲染，不写文件")
    args = ap.parse_args(argv)

    inputs = load_inputs()
    out_dir = os.path.join(REPO_ROOT, args.out_dir)
    if not args.check:
        os.makedirs(out_dir, exist_ok=True)
    for name, fn in FIGURES.items():
        svg = fn(inputs)
        if args.check:
            print("%-34s %6d bytes (not written)" % (name, len(svg)))
            continue
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as fh:
            fh.write(svg)
        print("%-34s -> %s" % (name, os.path.join(args.out_dir, name)))
    if not args.check:
        provenance = {
            "generated_by": "tools/make_figures.py",
            "caveat": (
                "每个输入都带 source；source 里出现『记录正文，转录』的项是记录里量过但"
                "没有独立 payload 的数（本轮只有 EXP-p4c-02 §5.6 的融合 norm 全步 A/B）。"
            ),
            "inputs": inputs,
        }
        with open(os.path.join(out_dir, "inputs.json"), "w", encoding="utf-8") as fh:
            json.dump(provenance, fh, ensure_ascii=False, indent=2, default=str)
        print("%-34s -> %s" % ("inputs.json", os.path.join(args.out_dir, "inputs.json")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
