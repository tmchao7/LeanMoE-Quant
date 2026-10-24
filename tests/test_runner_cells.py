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
    TOP1_MIN,
    _matrix_axes,
    _moe_for_exp,
    _weight_bytes_actual,
    build_exp_id,
    citation_source_for,
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
# ★ 新加轴时必须同时登记在这里：它是"骨架文件名 ↔ `build_exp_id`"的唯一真值来源测试
#   的解析器；漏登记会让那条测试报"缺轴取值"而不是静默放过（这正是它该有的行为）。
AXIS_KEYS = ("bit_width", "group_size", "dequant", "execution", "ctx", "attn")


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
    if not dirs:
        # results/ 是 gitignored 的机内工件：干净 clone（CI）里没有格子可复核。
        # 跳过而不是失败 —— 本测试的判别对象是"本地已有 payload 的机器"上的轴漂移。
        pytest.skip(
            "results/ 无带轴格子（干净 clone/CI）—— 本测试只在有本地 payload 的机器上有意义"
        )
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
    # EXP-p4c-01：attn 是**声明轴**（不是静态 config），顺序 = yaml 里 factors 的顺序
    assert _matrix_axes("EXP-p4c-01-in-graph-attention") == ["bit_width", "attn", "ctx"]


def test_attn_axis_is_declared_and_guarded():
    """★ `EXP-p4c-01` 的 `attn` 轴：两个臂必须是**不同的 cell id**，且缺值必须报错。

    第二条是防跑偏的关键：静默退回模型默认臂会写出一份"看起来是 kernel 证据"的 sdpa
    结果（`_execution_for_exp` 的注释里记着同类事故：臂选错了，报错说的却是网络）。
    """
    base = "EXP-p4c-01-in-graph-attention"
    sdpa = build_exp_id(base, {"bit_width": "w8a16", "attn": "sdpa", "ctx": 1024})
    kernel = build_exp_id(base, {"bit_width": "w8a16", "attn": "kernel", "ctx": 1024})
    assert sdpa != kernel
    assert sdpa.endswith("__bit_width=w8a16_attn=sdpa_ctx=1024")
    assert kernel.endswith("__bit_width=w8a16_attn=kernel_ctx=1024")
    with pytest.raises(ValueError, match="attn"):
        build_exp_id(base, {"bit_width": "w8a16", "ctx": 1024})


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
            "--rmsnorm",
            "hf",
        ]
    )
    out = capsys.readouterr()
    assert rc == 2
    assert "门实测" in out.err


def test_quantized_cudagraph_is_no_longer_refused_before_loading(capsys, monkeypatch):
    """P3（2026-10-07）：量化 × cudagraph 的"未接线"拒绝已删除。

    ★ 断言必须与机器无关：把 CUDA 判成不可用，这样"走到哪一道门"是可复现的
    （否则在有 GPU 的服务器上这个测试会真的去加载 13.4 GB 模型）。
    """
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
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
            # ★ 必须 --force：这一格在跑过的机器上 `is_done()` 为真，会被短路成
            #   `skip (done)` 返回 0 —— 断言就变成"本机跑没跑过这一格"的函数（实测红过）。
            "--force",
            "--rmsnorm",
            "hf",
        ]
    )
    out = capsys.readouterr()
    assert rc == 2
    assert "requires CUDA" in out.err, out.err
    assert "尚未接线" not in out.err, "P3 的接线拒绝应当已经删除"


def test_dry_run_surfaces_a_quantized_cudagraph_cell_with_its_cited_gate(capsys):
    """dry-run 必须答得出"这一格叫什么、门是什么状态" —— 量化 × cudagraph 也是。"""
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
            "--dequant",
            "separate",
            "--dry-run",
        ]
    )
    out = capsys.readouterr()
    assert rc == 0
    assert (
        "would run: EXP-p4b-01-truncation__bit_width=w8a16_execution=cudagraph_ctx=1024" in out.out
    )
    assert "gate: PASS" in out.out, out.out
    assert "execution=cudagraph" in out.out


def test_run_cudagraph_accepts_the_quant_cfg_hook():
    """P3 的接线点：`run_cudagraph` 必须真的收 `quant_cfg`（不是靠文档声称）。"""
    import inspect

    from benchmarks.runner import run_cudagraph

    sig = inspect.signature(run_cudagraph)
    assert "quant_cfg" in sig.parameters
    assert sig.parameters["quant_cfg"].default is None


# ---------------------------------------------------------------------------
# 门引用表：与 EXP-p1-04 的落盘工件对齐
# ---------------------------------------------------------------------------


def _artifact_for_row(bit_width: str, group_size: int, ctx: int) -> Path:
    """门行 -> 它引用的工件路径。真值来自 `runner.citation_source_for`（缺省按 ctx 推
    EXP-p1-04 的 5 窗规范件；w8a8@4096 自 R3 裁决起引 10 窗复测件）。"""
    src = citation_source_for(bit_width, group_size, ctx)
    assert src is not None, "ctx=%d 没有已知的门引用工件" % ctx
    return RESULTS / src["artifact"]


def _top1_mean(arm: dict) -> float:
    """门的口径是 top-1 的**均值**（p1-04 记录：mean_pass），不是逐窗口最小值。

    ★ 优先读工件自己算的 `aggregate.top1_mean`（`tools/quant_probe.aggregate_windows`
    现在会写它）；老工件没有这个键时**退回逐窗口重算**，这样新旧工件都能钉。
    按下限判门会否掉 w8a16（它有一个窗口 0.9394 < 0.95），与已记录的 PASS 矛盾。
    """
    agg = arm.get("aggregate") or {}
    if agg.get("top1_mean") is not None:
        return float(agg["top1_mean"])
    vals = [
        v["top1_agreement"]
        for k, v in arm["windows"].items()
        if k.startswith("w") and v.get("top1_agreement") is not None
    ]
    return sum(vals) / len(vals)


def test_gate_citations_match_the_artifacts():
    """`GATE_CITATIONS` 的 kl / top-1 必须**逐字**来自其引用工件的落盘实测。

    代码里写死的门值若与实测漂移，规则 2 就会在错误的配置上开绿灯 —— 这条测试
    把代码钉在 artifacts 上（"哪行引哪张工件"由 `citation_source_for` 唯一裁决）。
    工件不在（fresh clone）时跳过：results/ 不进 git。
    """
    rows = [(key, cited, citation_source_for(*key)) for key, cited in GATE_CITATIONS.items()]
    missing = [
        key for key, _, src in rows if src is None or not (RESULTS / src["artifact"]).exists()
    ]
    if missing:
        pytest.skip("引用工件不在本机（results/ 是 gitignored）：缺 %s" % missing)
    for (bit_width, group_size, ctx), cited, src in rows:
        artifact = json.loads((RESULTS / src["artifact"]).read_text(encoding="utf-8"))
        arm = artifact["arms"][src["arm"]]
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


def test_artifact_derives_the_same_verdict_as_the_runner():
    """★ 工件自己必须能回答"这个臂过没过门"，且与 runner 的门**逐格一致**。

    以前 `tools/quant_probe.aggregate_windows` 只写 `top1_min`、不写 `top1_mean`，
    而 `rec["gate"]` 用项目明确否掉的"下限"判据 —— 于是 w8a16@1024 在工件里
    是 `top1_min_pass=false`，runner 却发 PASS。引用值只能手抄进 `GATE_CITATIONS`。
    现在 `aggregate_windows` 同时写 `top1_mean` 与由均值推出的 `verdict`，
    这条测试用**工具自己的代码**重算一遍并逐格对齐。
    """
    import copy

    from tools.quant_probe import aggregate_windows

    checked = 0
    for (bit_width, group_size, ctx), cited in GATE_CITATIONS.items():
        src = citation_source_for(bit_width, group_size, ctx)
        if src is None:
            continue
        artifact_path = RESULTS / src["artifact"]
        if not artifact_path.exists():
            continue
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        arm_id = src["arm"]
        if arm_id not in artifact.get("arms", {}):
            continue
        arm = copy.deepcopy(artifact["arms"][arm_id])
        aggregate_windows({"arms": {arm_id: arm}}, kl_gate_w8=0.01, kl_gate_w4=0.05, top1_gate=0.95)
        top1_mean = arm["aggregate"]["top1_mean"]
        assert top1_mean is not None, "%s 的工件算不出 top-1 均值" % arm_id
        assert abs(top1_mean - cited["top1"]) < 5e-7, "%s@%d 工件 top-1 != 引用" % (
            bit_width,
            ctx,
        )
        expected = gate_from_metrics(bit_width, arm["aggregate"]["kl_mean"], top1_mean)
        assert arm["gate"]["verdict"] == expected, "%s@%d 工件 verdict != runner 门" % (
            bit_width,
            ctx,
        )
        checked += 1
    if not checked:
        pytest.skip("EXP-p1-04 的工件不在本机（results/ 是 gitignored）")


def test_the_gate_is_a_conjunction_not_decoration():
    """门是**合取**：KL 过但 top-1 不过 ⇒ FAIL；只看 KL 会误判为 PASS。

    ★ 2026-10-08（P2i）：这条原来读的是**当时的** `w8a8@4096` 引用（v1：KL 0.004606 过、
    top-1 0.9455 不过）。换实现后同一个臂的 top-1 变成 0.9636 ⇒ 那格不再"咬人"。
    判据本身没错，错的是把"逻辑"钉在一格贴线的实测上 —— 所以现在用**构造的实测值**
    测逻辑；真实数据的贴线事实由下面那条留档测试负责。
    """
    assert gate_from_metrics("w8a8", 0.004, 0.90) == "FAIL", "KL 过、top-1 不过必须 FAIL"
    assert gate_from_metrics("w8a8", 0.02, 1.0) == "FAIL", "KL 不过必须 FAIL"
    assert gate_from_metrics("w8a16", 0.004, 0.99) == "PASS"
    assert gate_from_metrics("w4a16", 0.049, 0.96) == "PASS", "w4 用 0.05 门槛"
    assert gate_from_metrics("w4a16", 0.051, 0.96) == "FAIL"


def test_w8a8_at_4096_straddles_the_top1_threshold_across_implementations():
    """★ 留档（P2i）：`w8a8@4096` 的 top-1 骑在门线上，**两个同样合法的实现跨在两侧**。

    v1（HF 的 `OlmoeRMSNorm` 原链）：top-1 0.9455 → FAIL
    v2（本仓库融合 RMSNorm）      ：top-1 0.9636 → PASS

    差 3/165 个位置；而且 v2 还改变了 fp16 参考的贪心轨迹（w2/w4）。
    所以"w8a8@4096 过不了 top-1 门"这个曾经被当成硬边界的结论，应当降级为
    **"在这个测法下不可分辨"**。这条测试的作用是防止有人把归档的 v1 悄悄删掉 ——
    两个工件都在，才能看出这个翻转是"贴线"而不是"改判"。
    """
    v1 = RESULTS / "EXP-p1-04-prompt-set-gate" / "probe_prompt_set_ctx4096_v1_hf_norm.json"
    v2 = RESULTS / "EXP-p1-04-prompt-set-gate" / "probe_prompt_set_ctx4096.json"
    if not v1.exists() or not v2.exists():
        pytest.skip("两个版本的工件不齐（results/ 是 gitignored）")
    a1 = json.loads(v1.read_text(encoding="utf-8"))["arms"]["w8a8_attn_experts_g128"]
    a2 = json.loads(v2.read_text(encoding="utf-8"))["arms"]["w8a8_attn_experts_g128"]
    t1, t2 = _top1_mean(a1), _top1_mean(a2)
    assert t1 < TOP1_MIN <= t2, (t1, t2)
    # 位置级别的差距极小（165 个位置里 3 个）：这正是"贴线"而不是"改判"的证据
    assert abs(t2 - t1) * 165 <= 4, (t1, t2)
    # ★ R3 裁决（2026-10-25）后引用表填的是 10 窗复测的 top-1 —— 与 5 窗的 t2 同值
    # （都是 0.963636），但判定由 KL 均值（0.0262 > 0.01）翻成 FAIL（de-publish）。
    assert abs(gate_for_cell("w8a8", 128, 4096)["top1"] - t2) < 5e-7


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
    assert abs(w8["kl"] - 0.001361) < 1e-9
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


def test_correctness_for_cell_unknown_ctx_never_falls_back_to_1024_artifact(monkeypatch):
    """引用工件映射只认 ctx ∈ {1024, 4096}（2026-10-10 评审补的闸）：
    若引用表将来出现新 ctx，不得静默套用 1024 的工件路径。"""
    from benchmarks import runner as runner_mod

    monkeypatch.setattr(
        runner_mod,
        "gate_for_cell",
        lambda *a, **k: {"gate": "PASS", "kl": 0.001, "top1": 0.99},
    )
    out = correctness_for_cell(
        {"bit_width": "w8a16", "group_size": 128, "ctx": 2048}, consistency_passed=True
    )
    assert out["gate"] == "FAIL"
    assert "2048" in out["thresholds"]["note"]
    assert out["kl"] is None


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


@pytest.fixture
def tiny_olmoe():
    """★ function scope，不是 module。

    以前是 `scope="module"`，而 `test_quantize_for_cfg_replaces_and_model_still_forwards`
    会**原地量化**这个模型（`quantize_model` 就是把 nn.Linear 换成 QuantLinear）。
    于是它之后的任何测试拿到的都是一个已经满身 QuantLinear 的"fp16"模型 ——
    只是恰好因为文件里 fp16 no-op 那条排在前面才没炸。任何重排、或新增一条
    "拿 tiny_olmoe 当未量化模型用"的测试，都会在离原因很远的地方失败。
    """
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
            "--rmsnorm",
            "hf",
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


def test_quantized_grouped_4bit_is_now_allowed_as_evidence(capsys):
    """★ 2026-10-07：4-bit 的 nibble 解包落地 → 量化 grouped 不再只做 8-bit。

    但 `w4a16 / g128` 的**引用门 FAIL**（KL 0.0933 > 0.05）：它照跑，只是
    `verdict=FAIL`、延迟只进 `_measurement.gate_failed_evidence`。dry-run 必须能看见这一格。
    """
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
            "--dry-run",
        ]
    )
    out = capsys.readouterr()
    assert rc == 0
    assert "would run: EXP-p4b-01-truncation__bit_width=w4a16_execution=eager_ctx=1024" in out.out
    assert "gate: FAIL" in out.out


def test_quantized_grouped_odd_group_size_is_refused(capsys, monkeypatch):
    """nibble 成对打包要求 group_size 偶数 —— 这条加载前拒绝仍然在。"""
    rc = runner_main(
        [
            "--exp",
            "EXP-p4b-01-truncation",
            "--bit-width",
            "w8a16",
            "--group-size",
            "127",
            "--execution",
            "eager",
            "--ctx",
            "1024",
            "--force",
            "--rmsnorm",
            "hf",
        ]
    )
    assert rc == 2
    # ★ 原来写的是 `"偶数" in out.err or "门实测" in out.err` —— 127 同时满足
    #   "奇数" 和 "没测过门"，而"没测过门"的拒绝排在前面，所以只要后者生效这条就绿，
    #   "奇数 group_size 必须拒绝" 这条规则从来没被独立验证过（把 nibble 偶数检查
    #   整段删掉，测试仍然绿）。所以这里 monkeypatch 掉门检查，单独钉它。
    from benchmarks import runner as _runner

    monkeypatch.setattr(
        _runner,
        "gate_for_cell",
        lambda *a, **k: {
            "kl": 0.0,
            "top1": 1.0,
            "rmsnorm": "hf",
            "gate": "PASS",
            "fake": "monkeypatched for this test",
        },
    )
    rc2 = runner_main(
        [
            "--exp",
            "EXP-p4b-01-truncation",
            "--bit-width",
            "w8a16",
            "--group-size",
            "127",
            "--execution",
            "eager",
            "--ctx",
            "1024",
            "--force",
            "--rmsnorm",
            "hf",
        ]
    )
    out2 = capsys.readouterr()
    assert rc2 == 2
    assert "偶数" in out2.err, "门检查被绕过之后，奇数 group_size 仍必须被拒绝"


def test_dry_run_surfaces_the_in_graph_attention_cells(capsys):
    """新实验（EXP-p4c-01）的格子必须能被 `--dry-run` 看见，并带上引用门的状态。"""
    rc = runner_main(
        [
            "--exp",
            "EXP-p4c-01-in-graph-attention",
            "--bit-width",
            "w8a16",
            "--attn",
            "kernel",
            "--ctx",
            "1024",
            "--dry-run",
        ]
    )
    out = capsys.readouterr()
    assert rc == 0
    assert (
        "would run: EXP-p4c-01-in-graph-attention__bit_width=w8a16_attn=kernel_ctx=1024" in out.out
    )
    assert "gate: PASS" in out.out, out.out


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


# ---------------------------------------------------------------------------
# P3：量化 × cudagraph 的 payload（**纯函数**，离线可测）
#
# 这两件事原来都是错的，且都会写出**违反硬规则 2** 的记录：
#   1. `config.bit_width` 硬编码 "fp16" → 量化格以 fp16 的名义落盘；
#   2. 门只用图回放门 → w8a8 会被跳过的引用门放行。
# ---------------------------------------------------------------------------
def _cudagraph_res(*, kl=0.0, top1=1.0, p50=4.0, weight_bytes_actual=6_600_000_000):
    """`build_cudagraph_payload` 需要的最小 res（数值全是自造，只为走代码路径）。"""
    from benchmarks.bytes import ModelSpec
    from benchmarks.timing import summarize

    steps = summarize([p50] * 20)
    spec = ModelSpec(
        name="OLMoE-test",
        layers=16,
        hidden=2048,
        num_attention_heads=16,
        num_key_value_heads=16,
        head_dim=128,
        num_experts=64,
        top_k=8,
        expert_ffn_dim=1024,
        vocab_size=50304,
        tie_word_embeddings=False,
    )
    budget = {
        "W": {"total_bytes": 6_000_000_000, "items": []},
        "T": {"total_bytes": 6_400_000_000},
        "KV": {"read_bytes": 100},
    }
    return {
        "spec": spec,
        "steps": steps,
        "ttft": {"p50": 60.0, "p99": 61.0, "samples": [60.0] * 5},
        "gap": {
            "gpu_p50_ms": 3.9,
            "gpu_busy_ms": 3.8,
            "host_p50_ms": 0.1,
            "host_share_pct": 2.0,
            "kernels_per_call": 900,
            "host_share_pct_residual": 1.0,
            "caveats": None,
        },
        "vram": {
            "weights_gib": 6.6,
            "kv_gib": 0.1,
            "activations_gib": 0.2,
            "workspace_gib": 0.3,
            "peak_total_gib": 7.2,
            "method": "m",
            "note": "n",
        },
        "breakdown": {
            "gpu_total_ms_per_step": 3.8,
            "kernels_per_step": 900,
            "bucket_ms": {"cudagraph_replay": 3.8},
            "bucket_share_pct": {"cudagraph_replay": 100.0},
            "top_kernels": [],
        },
        "budget": budget,
        "weight": {"total_bytes": 6_000_000_000},
        "attainment": {
            "floor_w_ms": 3.8,
            "floor_t_ms": 4.1,
            "quant_attainment_pct": 95.0,
            "bw_attainment_pct": 100.0,
            "denominator_note": "n",
        },
        "measured_bw_gbps": 1561.9,
        "weight_bytes_actual": weight_bytes_actual,
        "correctness": {
            "kl": kl,
            "top1_agreement": top1,
            "seed_token_match": True,
            "token_sequence_match": True,
            "eager_seed_token": 1,
            "graph_seed_token": 1,
            "eager_output_tokens": [1, 2],
            "graph_output_tokens": [1, 2],
            "informational_vs_hf_dynamic_cache": {"kl": 1.7e-4, "top1_agreement": 1.0},
        },
        "capture": {"capture_success": True},
        "moe": "grouped",
        "attn": "sdpa",
        "moe_blocks_replaced": 16,
        "dequant_arm": None,
        "dequant_activity": None,
        "quantize_report": None,
    }


def _cg_cfg(bit_width="fp16", dequant="none"):
    return {
        "bit_width": bit_width,
        "dequant": dequant,
        "group_size": 128 if bit_width != "fp16" else None,
        "execution": "cudagraph",
        "moe": "grouped",
        "attn": "sdpa",
        "ctx": 1024,
        "batch": 1,
        "max_new_tokens": 32,
        "warmup": 5,
        "timed": 20,
        "dtype": "float16",
        "fusion": "none",
    }


def test_cudagraph_payload_fp16_keeps_its_historical_shape():
    """fp16 的 cudagraph 格口径**逐字不变**：kl/top-1 就是图回放那对数字。"""
    from benchmarks.runner import build_cudagraph_payload
    from benchmarks.schema import validate_result

    res = _cudagraph_res(kl=0.0, top1=1.0)
    payload = build_cudagraph_payload(
        "EXP-p0-03-baseline-cudagraph__ctx=1024",
        res,
        _cg_cfg("fp16"),
        {"statement": "s", "refutation_signal": "r"},
    )
    assert payload["config"]["bit_width"] == "fp16"
    assert payload["correctness"]["kl"] == 0.0
    assert payload["correctness"]["gate"] == "PASS"
    assert payload["latency"]["per_token_p50_ms"] == 4.0
    assert validate_result(payload) == []


def test_cudagraph_payload_takes_bit_width_and_the_cited_gate_from_cfg():
    """★ P3 的核心修复：量化格的 `bit_width` 来自 cfg，门 = 引用门 ∧ 图回放门。"""
    from benchmarks.runner import build_cudagraph_payload
    from benchmarks.schema import validate_result

    res = _cudagraph_res(kl=0.0, top1=1.0)
    res["dequant_arm"] = "fused"
    res["quantize_report"] = {
        "spec": {"dequant": "fused"},
        "replaced_per_target": {"attention": 64},
    }
    res["dequant_activity"] = {"n_quant_linear": 64, "fused_calls": 3840, "fallbacks": {}}
    payload = build_cudagraph_payload(
        "EXP-p4b-01-truncation__bit_width=w8a16_execution=cudagraph_ctx=1024",
        res,
        _cg_cfg("w8a16", "fused"),
        {"statement": "s", "refutation_signal": "r"},
    )
    assert payload["config"]["bit_width"] == "w8a16", "硬编码 fp16 的坑必须已修"
    assert payload["config"]["dequant"] == "fused"
    assert payload["config"]["group_size"] == 128
    # KL 是**引用** p1-04 的实测值，不是图回放那对数字
    assert payload["correctness"]["kl"] == GATE_CITATIONS[("w8a16", 128, 1024)]["kl"]
    assert payload["correctness"]["gate"] == "PASS"
    assert payload["correctness"]["consistency"]["passed"] is True
    assert payload["correctness"]["graph_replay"]["kl"] == 0.0
    assert payload["latency"]["per_token_p50_ms"] == 4.0
    assert payload["_measurement"]["dequant"]["arm_recorded"] == "fused"
    assert validate_result(payload) == []


def test_cudagraph_payload_does_not_let_a_failed_citation_through():
    """w8a8 的引用门 FAIL → gate 必须 FAIL、latency 全 null（硬规则 2），即使图是忠实的。"""
    from benchmarks.runner import build_cudagraph_payload
    from benchmarks.schema import validate_result

    res = _cudagraph_res(kl=0.0, top1=1.0)
    payload = build_cudagraph_payload(
        "EXP-p4b-01-truncation__bit_width=w8a8_execution=cudagraph_ctx=1024",
        res,
        _cg_cfg("w8a8", "separate"),
        {"statement": "s", "refutation_signal": "r"},
    )
    assert payload["correctness"]["gate"] == "FAIL"
    assert payload["verdict"]["verdict"] == "FAIL"
    assert payload["latency"]["per_token_p50_ms"] is None
    assert payload["_measurement"]["gate_failed_evidence"]["per_token_p50_ms"] == 4.0
    assert validate_result(payload) == []


def test_cudagraph_payload_fails_when_the_graph_is_not_faithful():
    """图回放门本身也要算数：图与 eager 不一致时，量化格同样 FAIL（合取的另一半）。"""
    from benchmarks.runner import build_cudagraph_payload

    res = _cudagraph_res(kl=0.5, top1=0.5)  # 图 vs eager 不忠实
    payload = build_cudagraph_payload(
        "EXP-p4b-01-truncation__bit_width=w8a16_execution=cudagraph_ctx=1024",
        res,
        _cg_cfg("w8a16", "separate"),
        {"statement": "s", "refutation_signal": "r"},
    )
    assert payload["correctness"]["gate"] == "FAIL"
    assert payload["correctness"]["consistency"]["passed"] is False
    assert payload["latency"]["per_token_p50_ms"] is None


def test_fused_rmsnorm_is_refused_before_loading_but_hf_is_not(capsys, monkeypatch):
    """★ P2i 的第六条拒绝：`--rmsnorm fused` 在本机跑不了时必须在**加载模型之前**拒绝。

    注入"triton 不可用"，这样断言与机器无关（服务器上有 triton，直接跑会真加载模型）。
    第二段证明 `--rmsnorm hf` **不撞**这条 —— 它是显式退路，不是被这条拒绝挡住的。
    """
    import torch

    import lemoq.kernels.rmsnorm as rmsnorm

    monkeypatch.setattr(rmsnorm, "unavailable_reason", lambda: "triton 不可用（测试注入）")
    # ★ 第二段（`--rmsnorm hf`）在有 GPU 的机器上会**真的加载模型跑这一格**（实测：
    #   服务器上 3 分钟的模型加载 + 占用 GPU）。把 CUDA 判成不可用，"走到哪一道门"就
    #   与机器无关了 —— 这也是本文件里其它拒绝测试的做法。
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    base = [
        "--exp",
        "EXP-p4c-01-in-graph-attention",
        "--bit-width",
        "w8a16",
        "--execution",
        "cudagraph",
        "--ctx",
        "1024",
        "--attn",
        "kernel",
        "--force",
    ]
    rc = runner_main([*base, "--rmsnorm", "fused"])
    out = capsys.readouterr()
    assert rc == 2, (rc, out.err)
    assert "rmsnorm" in out.err and "测试注入" in out.err, out.err

    rc_hf = runner_main([*base, "--rmsnorm", "hf"])
    out_hf = capsys.readouterr()
    assert "rmsnorm" not in out_hf.err, out_hf.err
    assert rc_hf == 2 and "requires CUDA" in out_hf.err, (rc_hf, out_hf.err)


def test_dry_run_surfaces_the_rmsnorm_implementation(capsys):
    """★ P2i：cell id 里**没有** norm 轴，所以 dry-run（唯一的只读查询）必须报出实现名 ——
    否则"这一格会用什么 norm"在跑之前无从得知，事后也只能翻 payload。

    dry-run 不加载模型、不需要 GPU/ triton，所以这条断言与机器无关。
    """
    rc = runner_main(
        [
            "--exp",
            "EXP-p4c-01-in-graph-attention",
            "--bit-width",
            "w8a16",
            "--execution",
            "cudagraph",
            "--ctx",
            "1024",
            "--attn",
            "kernel",
            "--dry-run",
        ]
    )
    out = capsys.readouterr()
    assert rc == 0, out.err
    assert "rmsnorm=fused" in out.out, out.out

    rc_hf = runner_main(
        [
            "--exp",
            "EXP-p4c-01-in-graph-attention",
            "--bit-width",
            "w8a16",
            "--execution",
            "cudagraph",
            "--ctx",
            "1024",
            "--attn",
            "kernel",
            "--rmsnorm",
            "hf",
            "--dry-run",
        ]
    )
    out_hf = capsys.readouterr()
    assert rc_hf == 0 and "rmsnorm=hf" in out_hf.out, out_hf.out


def test_rmsnorm_fields_reach_the_payload_from_both_config_blocks():
    """★ P2i 的教训：payload 里有**两处** `payload["config"] = {…}`（eager / cudagraph），
    后者会覆盖前者。把字段写成一个函数、两处都展开，才不会"加在会被覆盖的那一块"
    （实测：v3 轮第一遍的单元格 payload 里 `config.rmsnorm` 是 None）。

    这条测试同时钉住三件事：默认值、实现名优先于 cfg、以及 replaced 计数。
    """
    from benchmarks.runner import _rmsnorm_payload_fields

    assert _rmsnorm_payload_fields({"rmsnorm": "fused", "rms_norms_replaced": 65}, {}) == {
        "rmsnorm": "fused",
        "rms_norms_replaced": 65,
    }
    # res 缺字段时退回 cfg；cfg 也缺时退回 "hf"（历史 payload 没有这个轴）
    assert _rmsnorm_payload_fields({}, {"rmsnorm": "fused"})["rmsnorm"] == "fused"
    assert _rmsnorm_payload_fields({}, {})["rmsnorm"] == "hf"
