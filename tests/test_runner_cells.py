"""test_runner_cells — runner 的量化格接线（**全部离线可测**）。

这一组测试钉住 `EXP-p2-01` / `EXP-p4b-01` 上卡前必须正确的几件事：

    1. cell id 必须由**矩阵声明的轴**拼出，且与 `results/` 里已有的骨架逐字相同
       —— 手拼 `__ctx=%d` 会写出对不上的文件名，`is_done()` 永远判不了；
    2. **拒绝**必须在任何模型加载之前失败，这样无 GPU 机器上也能验证：
       fused 臂在本机不可用（没有 triton/CUDA）/ 没有门实测 / 量化+cudagraph 未接线；
    3. `GATE_CITATIONS` 的数值必须与 `EXP-p1-04` 的落盘工件一致（代码与实测对齐）；
    4. 量化 hook 在 CPU tiny OLMoE 上可跑：QuantLinear 的 packed/scale 是 **buffer**，
       而 `weight_bytes_actual` 只数 parameters —— 这是量化格字节账的头号陷阱；
    5. **硬规则 2 的机械执行点**：`gate == FAIL` 的格子照跑，但 `latency` 段必须全 null
       （实测原始值进 `_measurement.gate_failed_evidence`）。

不测什么：延迟数字（GPU）。fused kernel 的**数值**门是
`tests/test_kernel_parity.py::test_triton_fused_gemm_matches_reference`（`-m gpu`）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import requires_torch

pytestmark = requires_torch

from benchmarks.runner import (  # noqa: E402
    GATE_CITATIONS,
    KL_MAX_W8,
    TOP1_MIN,
    _matrix_axes,
    _moe_for_exp,
    _weight_bytes_actual,
    build_exp_id,
    correctness_for_cell,
    gate_for_cell,
    gate_from_metrics,
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
    # ★ E7 两臂必须同一 MoE 路径（矩阵声明 grouped；hf 不能 capture，p0-03 blocked）
    assert "moe=grouped" in out.out


def test_dry_run_answers_even_when_the_cell_is_already_done(capsys, tmp_path):
    """`--dry-run` 是只读查询，**不受 `is_done` 短路**。

    `results/` 是机器本地的（gitignored）：同一格在 GPU 机器上跑过之后，dry-run 若被
    短路成 `skip (done)`，测试与人就都问不到"这一格叫什么、门什么状态"。
    这条在服务器上**实测红过**（本地 Mac 的 results/ 干净，所以看不到）。
    """
    cell = "EXP-p4b-01-truncation__bit_width=fp16_execution=eager_ctx=1024"
    d = tmp_path / cell
    d.mkdir()
    (d / "result.json").write_text(json.dumps({"status": "done"}), encoding="utf-8")
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
            "--results-dir",
            str(tmp_path),
            "--dry-run",
        ]
    )
    out = capsys.readouterr()
    assert rc == 0
    assert "would run: %s" % cell in out.out, "dry-run 被 is_done 短路了"
    assert "moe=grouped" in out.out


def test_fused_dequant_is_refused_where_the_kernel_cannot_run(capsys, monkeypatch):
    """real run（非 dry-run）必须在**加载模型之前**拒绝 fused —— 这台机器跑不了它。

    ★ 2026-10-07 起 kernel 已落地（`src/lemoq/kernels/_dequant_triton.py`），
      所以拒绝的理由从"kernel 未实现"变成"本机没有 triton/CUDA"。测试 monkeypatch
      可用性判断，这样在有 GPU 的服务器上也是同一份断言（否则它会真去加载模型）。
    """
    import benchmarks.runner as runner_mod

    monkeypatch.setattr(runner_mod, "fused_unavailable_reason", lambda: "triton 不可用（测试注入）")
    rc = runner_main(
        ["--exp", "EXP-p2-01-dequant-fusion", "--bit-width", "w8a16", "--dequant", "fused"]
    )
    out = capsys.readouterr()
    assert rc == 2
    assert "拒绝" in out.err and "fused" in out.err and "测试注入" in out.err


def test_fused_dequant_on_fp16_is_refused(capsys, monkeypatch):
    """fp16 没有反量化语义：`--dequant fused --bit-width fp16` 是配置错误，不是实验格。"""
    import benchmarks.runner as runner_mod

    monkeypatch.setattr(runner_mod, "fused_unavailable_reason", lambda: None)
    rc = runner_main(
        ["--exp", "EXP-p2-01-dequant-fusion", "--bit-width", "fp16", "--dequant", "fused"]
    )
    out = capsys.readouterr()
    assert rc == 2
    assert "只在量化格" in out.err


def test_fused_availability_check_is_none_when_the_kernel_is_usable(monkeypatch):
    """`fused_unavailable_reason()` 可用时必须返回 None（runner 的拒绝依赖它）。"""
    import benchmarks.runner as runner_mod
    import lemoq.kernels._dequant_triton as k

    monkeypatch.setattr(k, "unavailable_reason", lambda: None)
    assert runner_mod.fused_unavailable_reason() is None


def test_latency_section_nulls_performance_when_the_gate_failed():
    """硬规则 2：gate FAIL 的格子不得在 latency 段报性能数字（schema 校验器的对偶）。"""
    from benchmarks.runner import latency_section_for_gate

    measured = {
        "ttft_p50_ms": 1.0,
        "ttft_p99_ms": 2.0,
        "per_token_p50_ms": 3.0,
        "per_token_p99_ms": 4.0,
        "tok_per_s": 5.0,
        "samples": {"per_token_ms": [3.0]},
    }
    # PASS：原样保留
    latency, evidence = latency_section_for_gate(True, measured)
    assert latency["per_token_p50_ms"] == 3.0 and evidence is None
    # FAIL：latency 全 null，原始值进证据块（不是删掉 —— 删掉等于"没测过"）
    latency, evidence = latency_section_for_gate(False, measured)
    assert latency["per_token_p50_ms"] is None and latency["tok_per_s"] is None
    assert latency["samples"] == {}
    assert evidence["per_token_p50_ms"] == 3.0 and "硬规则 2" in evidence["note"]


def test_schema_rejects_the_payload_shape_this_policy_prevents():
    """把上面那条策略与 `validate_result` 对上：报性能 + gate FAIL 会被判违规。"""
    from benchmarks.schema import empty_result, validate_result

    payload = empty_result("EXP-p2-01-dequant-fusion__bit_width=w4a16_dequant=fused")
    payload["correctness"]["gate"] = "FAIL"
    payload["latency"]["per_token_p50_ms"] = 3.0
    assert any("硬规则 2" in p for p in validate_result(payload))
    # 策略产出的形状必须通过校验
    from benchmarks.runner import latency_section_for_gate

    payload["latency"] = latency_section_for_gate(
        False, dict(payload["latency"], per_token_p50_ms=3.0)
    )[0]
    assert validate_result(payload) == []


def test_quant_spec_maps_dequant_axis_and_keeps_legacy_none(capsys):
    """`--dequant` 进 spec；老的 `dequant="none"`（p0/p4b 的量化格）等价于 separate。"""
    fused = quant_spec_for_cfg({"bit_width": "w4a16", "group_size": 128, "dequant": "fused"})
    assert fused.dequant == "fused" and fused.weight_bits == 4
    legacy = quant_spec_for_cfg({"bit_width": "w8a16", "group_size": 128, "dequant": "none"})
    assert legacy.dequant == "separate"
    missing = quant_spec_for_cfg({"bit_width": "w8a16", "group_size": 128})
    assert missing.dequant == "separate"


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


def _artifact_for_ctx(ctx: int) -> Path:
    """CTX -> 那张工件。P2（EXP-p4b-01 §3）把门补测到了 ctx=4096，两张工件同格式。"""
    name = "probe_prompt_set.json" if int(ctx) == 1024 else "probe_prompt_set_ctx4096.json"
    return RESULTS / "EXP-p1-04-prompt-set-gate" / name


def _top1_mean(arm: dict) -> float:
    """工件的 aggregate 只存 top1_min；门的口径是**均值**（p1-04 记录：mean_pass）。"""
    vals = [
        v["top1_agreement"]
        for k, v in arm["windows"].items()
        if k.startswith("w") and v.get("top1_agreement") is not None
    ]
    return sum(vals) / len(vals)


def test_gate_citations_match_the_artifacts():
    """`GATE_CITATIONS` 的 kl / top-1 必须**逐字**来自 EXP-p1-04 的落盘工件。

    代码里写死的门值若与实测漂移，规则 2 就会在错误的配置上开绿灯 —— 这条测试
    把代码钉在 artifacts 上。工件不在（fresh clone）时跳过：results/ 不进 git。
    """
    missing = [
        c for c in {ctx for (_, _, ctx) in GATE_CITATIONS} if not _artifact_for_ctx(c).exists()
    ]
    if missing:
        pytest.skip("EXP-p1-04 的工件不在本机（results/ 是 gitignored）：缺 ctx=%s" % missing)
    for (bit_width, group_size, ctx), cited in GATE_CITATIONS.items():
        artifact = json.loads(_artifact_for_ctx(ctx).read_text(encoding="utf-8"))
        arm = artifact["arms"]["%s_attn_experts_g%d" % (bit_width, group_size)]
        kl_mean = arm["aggregate"]["kl_mean"]
        assert abs(kl_mean - cited["kl"]) < 5e-7, "%s@%d 引用 kl %r != 实测 %r" % (
            bit_width,
            ctx,
            cited["kl"],
            kl_mean,
        )
        top1_mean = _top1_mean(arm)
        assert abs(top1_mean - cited["top1"]) < 5e-7, "%s@%d 引用 top-1 %r != 实测 %r" % (
            bit_width,
            ctx,
            cited["top1"],
            top1_mean,
        )
        # ★ gate 必须能由**实测值 + METRICS 门槛**推出来（合取），不许手写
        expected = gate_from_metrics(bit_width, kl_mean, top1_mean)
        assert (
            gate_for_cell(bit_width, group_size, ctx)["gate"] == expected
        ), "%s@%d 的 gate 与实测/门槛不符" % (bit_width, ctx)


def test_the_top1_criterion_actually_bites_at_ctx4096():
    """★ P2 的关键发现：`w8a8@4096` 的 KL 过门、**top-1 不过门**。

    这条测试钉住"门是合取"而不是装饰：若有人把规则退回"只看 KL"，
    `w8a8@4096` 就会变成 PASS 并发表延迟 —— 那是放宽正确性门（硬规则 2）。
    """
    if not _artifact_for_ctx(4096).exists():
        pytest.skip("ctx=4096 的工件不在本机")
    cite = gate_for_cell("w8a8", 128, 4096)
    assert cite is not None
    assert cite["kl"] < KL_MAX_W8, "这条测试的前提是 KL 确实过门"
    assert cite["top1"] < TOP1_MIN, "这条测试的前提是 top-1 确实不过门"
    assert cite["gate"] == "FAIL"
    # 对照：只看 KL 会误判为 PASS —— 显式记录这个陷阱
    assert gate_from_metrics("w8a8", cite["kl"], 1.0) == "PASS"


def test_gate_for_cell_returns_none_for_unmeasured_configs():
    assert gate_for_cell("w8a16", 128, 1024) is not None
    assert gate_for_cell("w8a16", 128, 4096) is not None  # P2 已补测
    assert gate_for_cell("w8a16", 64, 1024) is None  # g64 没测过
    assert gate_for_cell("w8a16", 128, 2048) is None  # ctx=2048 没测过
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


def _fresh_olmoe():
    """新模型。★ 不要用共享 fixture 做**会改模型**的测试：
    量化测试会把 fixture 变成 QuantLinear 满身的模型，后面的测试拿到的是脏状态。"""
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


@pytest.fixture(scope="module")
def tiny_olmoe():
    return _fresh_olmoe()


def test_matrix_declares_one_moe_path_for_both_arms():
    """E7 的两臂必须同一 MoE 实现（硬规则 1）：矩阵声明 grouped，runner 读得到。"""
    assert _moe_for_exp("EXP-p4b-01-truncation") == "grouped"
    # 其它实验不受影响：没声明 config.moe 的仍然默认 hf
    assert _moe_for_exp("EXP-p0-02-baseline-fp16") == "hf"
    assert _moe_for_exp("EXP-not-in-matrix") == "hf"


def test_run_baseline_accepts_moe_parameter():
    """eager 臂必须能接 MoE 替换 —— 否则 E7 会一头 grouped 一头 hf（混轴）。"""
    import inspect

    from benchmarks.runner import run_baseline

    params = inspect.signature(run_baseline).parameters
    assert "moe" in params and params["moe"].default == "hf"
    # ★ P1 之后量化与 grouped MoE **可以**共存（experts 在融合张量上量化）；
    #   docstring 必须写明这一点，否则读者还以为两者互斥。
    doc = inspect.getdoc(run_baseline) or ""
    assert "量化 × grouped" in doc
    assert "P1" in doc


def test_quantized_over_non_hf_moe_is_refused(capsys):
    """★ P1 之后：**量化 × gather** 仍然不存在（gather 直接索引 fp16 融合张量）。

    旧版本这里用矩阵默认的 grouped —— P1 落地后那条路已经通了，再拿它当"应被拒绝"
    会真的去加载模型（实测：跑成 3 分钟的 huggingface 超时）。
    """
    rc = runner_main(
        [
            "--exp",
            "EXP-p4b-01-truncation",
            "--bit-width",
            "w8a16",
            "--execution",
            "eager",
            "--ctx",
            "1024",
            "--moe",
            "gather",
        ]
    )
    out = capsys.readouterr()
    assert rc == 2
    assert "拒绝" in out.err and "gather" in out.err
    # fp16 臂不受影响：同一命令只换 bit_width 就不该撞这条
    rc_fp16 = runner_main(
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
    assert rc_fp16 == 0


def test_quantized_grouped_4bit_is_refused(capsys):
    """量化 grouped 只实现 8-bit：4-bit 的 nibble 解包未实现 → 加载前拒绝。"""
    rc = runner_main(
        [
            "--exp",
            "EXP-p4b-01-truncation",
            "--bit-width",
            "w4a16",
            "--execution",
            "eager",
            "--ctx",
            "1024",
        ]
    )
    out = capsys.readouterr()
    assert rc == 2
    assert "4-bit" in out.err or "8-bit" in out.err


def test_weight_bytes_actual_counts_static_experts():
    """MoE 替换后字节账不许少算 96%：`_static_experts` 是普通张量，不在 parameters() 里。

    历史教训：EXP-p0-04 / p2-02 记的是 953,421,824 B（≈ 只剩 attention+embedding）。
    """
    import torch

    from lemoq.kernels.moe_static import replace_moe_blocks

    model = _fresh_olmoe()
    before = _weight_bytes_actual(model)
    assert before > 0
    # ★ impl 名是 "gather"：`static` 是 p0-04 时代的旧名，已不在 VALID_MOE_IMPLS
    #   （实测会 raise unknown impl；runner 里 static→gather 的归一化就是为这个）。
    #   历史教训：EXP-p0-04 / p2-02 记的 weight_bytes_actual 是 953,421,824 B，
    #   因为 _static_experts 不在 parameters() 里 —— 那两个格子已落盘，不改。
    replaced = replace_moe_blocks(model, impl="gather")
    assert replaced == 2, "tiny 模型有 2 层，每层一个 sparse MoE block"
    after = _weight_bytes_actual(model)
    # fuse 只是把 experts 的元素重排（gate‖up, down），元素数与字节数都应守恒
    assert after == before, "替换前后驻留权重字节应相等：%d != %d" % (after, before)
    with torch.no_grad():
        out = model(torch.tensor([[1, 2, 3, 4]]))
    assert out.logits.shape == (1, 4, 256), "替换后必须还能前向"


def test_weight_bytes_actual_counts_quantized_buffers():
    """量化后字节账不许少算 96%：QuantLinear 的 packed/scale 是 buffer。"""
    model = _fresh_olmoe()
    plain = _weight_bytes_actual(model)
    qreport = quantize_for_cfg(model, {"bit_width": "w8a16", "group_size": 32})
    assert qreport is not None
    counted = _weight_bytes_actual(model, qreport)
    # ★ 不断言与 plain 的比例：测试模型是 fp32、packed 是 int8，比例只由 dtype 决定；
    #   真实模型是 fp16，w8a16 的 49.4% 已由 EXP-p1-01 实测。
    #   钉住的不变量是文档里记的陷阱：**不传 qreport 就会漏掉 buffer**。
    without_buffers = _weight_bytes_actual(model, None)
    assert without_buffers < counted, "量化后不传 qreport 必须少算（buffer 不在 parameters 里）"
    assert without_buffers < plain, "量化吃掉了 linears 的参数（它们进了 buffer）"


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
