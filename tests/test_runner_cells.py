"""test_runner_cells — runner 的量化格接线（**全部离线可测**）。

这一组测试钉住 `EXP-p2-01` / `EXP-p4b-01` 上卡前必须正确的四件事：

    1. cell id 必须由**矩阵声明的轴**拼出，且与 `results/` 里已有的骨架逐字相同
       —— 手拼 `__ctx=%d` 会写出对不上的文件名，`is_done()` 永远判不了；
    2. 三条**拒绝**（fused kernel 未实现 / 没有门实测 / 量化+cudagraph 未接线）
       必须在任何模型加载之前失败，这样无 GPU 机器上也能验证；
    3. `GATE_CITATIONS` 的数值必须与 `EXP-p1-04` 的落盘工件一致（代码与实测对齐）；
    4. 量化 hook 在 CPU tiny OLMoE 上可跑：QuantLinear 的 packed/scale 是 **buffer**，
       而 `weight_bytes_actual` 只数 parameters —— 这是量化格字节账的头号陷阱。

不测什么：延迟数字（GPU）、融合 kernel（`src/lemoq/kernels/__init__.py` 记录了
"离线不写未编译 kernel"的决定，fused 臂在 kernel 落地前由 runner 拒绝）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import requires_torch

pytestmark = requires_torch

from benchmarks.runner import (  # noqa: E402
    GATE_CITATIONS,
    _matrix_axes,
    build_exp_id,
    correctness_for_cell,
    gate_for_cell,
    quant_spec_for_cfg,
    quantize_for_cfg,
)
from benchmarks.runner import (
    main as runner_main,
)

RESULTS = Path(__file__).resolve().parent.parent / "results"
# cell id 里出现过的轴键（解析已有目录名用；顺序无所谓，输出顺序由 _matrix_axes 定）
AXIS_KEYS = ("bit_width", "group_size", "dequant", "execution", "ctx")


def _parse_cell_dir(dirname: str) -> tuple[str, dict[str, str]] | None:
    """`EXP-x__bit_width=w8a8_dequant=fused` -> (`EXP-x`, {bit_width: w8a8, ...})。"""
    if "__" not in dirname:
        return None
    exp_id, suffix = dirname.split("__", 1)
    marks: list[tuple[int, str]] = []
    for key in AXIS_KEYS:
        pos = suffix.find(key + "=")
        while pos >= 0:
            if pos == 0 or suffix[pos - 1] in "_-":
                marks.append((pos, key))
            pos = suffix.find(key + "=", pos + 1)
    if not marks:
        return None
    marks.sort()
    values: dict[str, str] = {}
    for i, (pos, key) in enumerate(marks):
        start = pos + len(key) + 1
        if i + 1 < len(marks):
            # 键值对之间的分隔符是 `_`，它属于**分隔符**而不是上一个键的值
            # （不减掉它，"w4a16_" 会被 join 成 `w4a16__group_size` —— 实测红过）
            end = marks[i + 1][0] - 1
        else:
            end = len(suffix)
        values[key] = suffix[start:end]
    return exp_id, values


def test_every_recorded_cell_id_is_reproducible_from_the_matrix():
    """`results/` 里每一个带轴的格子，都必须能由 `build_exp_id` 复原。

    这是 cell id 的**唯一真值来源**测试：runner 与 harness 展开的是同一批文件名，
    任何一侧改了轴顺序或键名，这里就会红。
    """
    dirs = sorted(p.name for p in RESULTS.glob("EXP-*__*") if p.is_dir())
    assert dirs, "results/ 里没有任何带轴的格子 —— 这个测试就失去意义了"
    checked = 0
    for dirname in dirs:
        parsed = _parse_cell_dir(dirname)
        if parsed is None:
            continue
        exp_id, values = parsed
        assert build_exp_id(exp_id, values) == dirname, dirname
        checked += 1
    assert checked >= 20, "只复原了 %d 个格子，覆盖不够" % checked


def test_matrix_axes_are_declared_in_yaml_order():
    """轴顺序 = 骨架文件名里的顺序（不是字典序）。"""
    assert _matrix_axes("EXP-p4b-01-truncation") == ["bit_width", "execution", "ctx"]
    assert _matrix_axes("EXP-p2-01-dequant-fusion") == ["bit_width", "dequant"]
    assert _matrix_axes("EXP-p1-01-w8-w4-accuracy") == ["bit_width", "group_size"]
    assert _matrix_axes("EXP-p0-02-baseline-fp16") == ["ctx"]
    # 矩阵里声明了、但没有轴的实验：id 不带后缀
    assert _matrix_axes("EXP-p4-01-moe-routing-overhead") == []


def test_build_exp_id_keeps_legacy_format_for_matrix_unknown_experiments():
    """矩阵不认识的实验保留老的 `__ctx=%d` —— 无法区分"无轴"与"不在矩阵"。"""
    assert build_exp_id("EXP-not-in-matrix", {"ctx": 1024}) == "EXP-not-in-matrix__ctx=1024"
    # 矩阵认识、但无轴：不加后缀
    assert build_exp_id("EXP-p4-01-moe-routing-overhead", {"ctx": 1024}) == (
        "EXP-p4-01-moe-routing-overhead"
    )


# ---------------------------------------------------------------------------
# CLI：dry-run 的 id 输出 + 三条拒绝
# ---------------------------------------------------------------------------


def test_dry_run_reproduces_the_p201_cell_id(capsys):
    rc = runner_main(
        [
            "--exp",
            "EXP-p2-01-dequant-fusion",
            "--bit-width",
            "w8a16",
            "--dequant",
            "separate",
            "--dry-run",
        ]
    )
    out = capsys.readouterr()
    assert rc == 0
    assert "would run: EXP-p2-01-dequant-fusion__bit_width=w8a16_dequant=separate" in out.out
    assert "gate: PASS" in out.out, "w8a16 的门是 p1-04 实测过的 PASS，dry-run 应当报出来"


def test_dry_run_reports_gate_fail_for_gate_failing_configs(capsys):
    rc = runner_main(
        [
            "--exp",
            "EXP-p2-01-dequant-fusion",
            "--bit-width",
            "w4a16",
            "--dequant",
            "fused",
            "--dry-run",
        ]
    )
    out = capsys.readouterr()
    assert rc == 0
    # dry-run 不拒绝：它要展示"这一格叫什么、门是什么状态"
    assert "would run: EXP-p2-01-dequant-fusion__bit_width=w4a16_dequant=fused" in out.out
    assert "gate: FAIL" in out.out


def test_dry_run_p4b_cell_id_matches_skeleton(capsys):
    rc = runner_main(
        [
            "--exp",
            "EXP-p4b-01-truncation",
            "--bit-width",
            "fp16",
            "--execution",
            "eager",
            "--ctx",
            "1024",
            "--dry-run",
        ]
    )
    out = capsys.readouterr()
    assert rc == 0
    assert "would run: EXP-p4b-01-truncation__bit_width=fp16_execution=eager_ctx=1024" in out.out


def test_fused_dequant_is_refused_without_a_kernel(capsys):
    """real run（非 dry-run）必须在加载前拒绝 fused —— kernel 还不存在。"""
    rc = runner_main(
        ["--exp", "EXP-p2-01-dequant-fusion", "--bit-width", "w8a16", "--dequant", "fused"]
    )
    out = capsys.readouterr()
    assert rc == 2
    assert "拒绝" in out.err and "fused" in out.err


def test_quantized_config_without_gate_citation_is_refused(capsys):
    """没有 EXP-p1-04 实测的配置（g64 没测过）不许跑 —— 硬规则 2 的执行点。"""
    rc = runner_main(
        [
            "--exp",
            "EXP-p2-01-dequant-fusion",
            "--bit-width",
            "w8a16",
            "--group-size",
            "64",
            "--dequant",
            "separate",
        ]
    )
    out = capsys.readouterr()
    assert rc == 2
    assert "门实测" in out.err


def test_quantized_cudagraph_is_refused(capsys):
    rc = runner_main(
        [
            "--exp",
            "EXP-p4b-01-truncation",
            "--bit-width",
            "w8a16",
            "--execution",
            "cudagraph",
            "--ctx",
            "1024",
        ]
    )
    out = capsys.readouterr()
    assert rc == 2
    assert "cudagraph" in out.err


# ---------------------------------------------------------------------------
# 门引用表：与 EXP-p1-04 的落盘工件对齐
# ---------------------------------------------------------------------------


def test_gate_citations_match_the_p104_artifact():
    """`GATE_CITATIONS` 的数字必须**逐字**来自 EXP-p1-04 的工件。

    代码里写死的门值若与实测漂移，规则 2 就会在错误的配置上开绿灯 —— 这条测试
    把代码钉在 artifacts 上。工件不在（fresh clone）时跳过：results/ 不进 git。
    """
    path = RESULTS / "EXP-p1-04-prompt-set-gate" / "probe_prompt_set.json"
    if not path.exists():
        pytest.skip("EXP-p1-04 的工件不在本机（results/ 是 gitignored）")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    for (bit_width, group_size, ctx), cited in GATE_CITATIONS.items():
        arm = artifact["arms"]["%s_attn_experts_g%d" % (bit_width, group_size)]
        mean = arm["aggregate"]["kl_mean"]
        assert abs(mean - cited["kl"]) < 5e-7, "%s 引用值 %r != 实测 %r" % (
            bit_width,
            cited["kl"],
            mean,
        )
        expected_gate = "PASS" if mean < 0.01 else "FAIL"  # W8 门（p1-04 全是 W8/W4 分档）
        if bit_width == "w4a16":
            expected_gate = "PASS" if mean < 0.05 else "FAIL"
        assert cited["gate"] == expected_gate, bit_width


def test_gate_for_cell_returns_none_for_unmeasured_configs():
    assert gate_for_cell("w8a16", 128, 1024) is not None
    assert gate_for_cell("w8a16", 64, 1024) is None  # g64 没测过
    assert gate_for_cell("w8a16", 128, 4096) is None  # ctx=4096 没测过
    assert gate_for_cell("w4a8", 128, 1024) is None  # 没有这种 scheme


def test_correctness_for_cell_fp16_uses_consistency_only():
    cfg = {"bit_width": "fp16", "ctx": 1024}
    ok = correctness_for_cell(cfg, consistency_passed=True)
    assert ok["gate"] == "PASS"
    assert ok["kl"] is None, "fp16 的 KL 必须是 null —— 冒充 0 过门是同义反复"
    assert "自洽性" in ok["thresholds"]["note"]
    assert correctness_for_cell(cfg, consistency_passed=False)["gate"] == "FAIL"


def test_correctness_for_cell_quantized_cites_p104_and_respects_fail():
    w8 = correctness_for_cell(
        {"bit_width": "w8a16", "group_size": 128, "ctx": 1024}, consistency_passed=True
    )
    assert w8["gate"] == "PASS"
    assert abs(w8["kl"] - 0.001273) < 1e-9
    assert "引用" in w8["thresholds"]["note"] and "EXP-p1-04" in w8["thresholds"]["note"]

    w4 = correctness_for_cell(
        {"bit_width": "w4a16", "group_size": 128, "ctx": 1024}, consistency_passed=True
    )
    assert w4["gate"] == "FAIL", "引用门 FAIL 的格子不能被一致性检查洗白"

    # 自洽性挂了，连 PASS 的引用门也要 FAIL
    assert (
        w8
        and correctness_for_cell(
            {"bit_width": "w8a16", "group_size": 128, "ctx": 1024}, consistency_passed=False
        )["gate"]
        == "FAIL"
    )

    uncited = correctness_for_cell(
        {"bit_width": "w8a16", "group_size": 64, "ctx": 1024}, consistency_passed=True
    )
    assert uncited["gate"] == "FAIL"
    assert uncited["kl"] is None
    assert "没有" in uncited["thresholds"]["note"]


# ---------------------------------------------------------------------------
# 量化 hook：CPU tiny OLMoE
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_olmoe():
    from transformers import OlmoeConfig, OlmoeForCausalLM

    cfg = OlmoeConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    return OlmoeForCausalLM(cfg).eval()


def test_quant_spec_excludes_router_from_targets():
    """★ targets 必须是 `(attention, experts)`：p1-04 的门就是不含 router 测的。

    `SchemeSpec.from_flags` 的默认值多一个 router，而只量化 router 就能让 W8 的 KL
    动 14-74%（EXP-p1-01 实测）—— 用默认值等于换了一个没过门的配置。
    """
    spec = quant_spec_for_cfg({"bit_width": "w8a16", "group_size": 128})
    assert spec is not None
    assert spec.targets == ("attention", "experts")
    assert "router" not in spec.targets
    assert spec.weight_bits == 8 and spec.group_size == 128
    assert quant_spec_for_cfg({"bit_width": "fp16"}) is None


def test_quantize_for_cfg_fp16_is_a_noop(tiny_olmoe):
    before = sum(p.numel() for p in tiny_olmoe.parameters())
    assert quantize_for_cfg(tiny_olmoe, {"bit_width": "fp16"}) is None
    assert sum(p.numel() for p in tiny_olmoe.parameters()) == before


def test_quantize_for_cfg_replaces_and_model_still_forwards(tiny_olmoe):
    import torch

    from lemoq.quant.linear import QuantLinear

    report = quantize_for_cfg(tiny_olmoe, {"bit_width": "w8a16", "group_size": 32})
    assert report is not None, "量化 hook 应当返回 QuantizeReport"
    assert report.replaced.get("attention", 0) > 0 and report.replaced.get("experts", 0) > 0
    assert report.spec.get("targets") == ["attention", "experts"] or tuple(
        report.spec.get("targets") or ()
    ) == ("attention", "experts")

    linears = [m for m in tiny_olmoe.modules() if isinstance(m, QuantLinear)]
    assert linears, "没有 QuantLinear —— 替换没发生"
    # ★ 字节账陷阱的直接断言：量化权重在 **buffer** 里，parameters 数不到它
    buffers = {b for m in linears for b, _ in m.named_buffers()}
    assert "packed" in buffers and "scale" in buffers, sorted(buffers)
    assert not any(
        n == "weight" for m in linears for n, _ in m.named_parameters()
    ), "QuantLinear 还挂着 weight 参数 —— 那会让字节账把量化权重数两遍"

    with torch.no_grad():
        out = tiny_olmoe(torch.tensor([[1, 2, 3, 4]]))
    assert out.logits.shape == (1, 4, 256), "量化后的模型必须还能前向"
