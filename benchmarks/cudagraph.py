"""CUDA Graph feasibility helpers for the fp16 OLMoE decode path.

This module deliberately implements only the first feasibility gate:
continuous Transformers ``StaticCache`` storage plus a one-token CUDA Graph.
It does not implement paged attention, expert dispatch, quantization, or a
fallback to eager execution when capture/replay fails.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from benchmarks.schema import empty_result
from benchmarks.timing import summarize

VALID_EXECUTIONS = ("eager", "cudagraph")


def normalize_execution(value: str) -> str:
    """Validate the execution arm instead of silently falling back."""
    if value not in VALID_EXECUTIONS:
        raise ValueError("unknown execution %r; expected one of %s" % (value, VALID_EXECUTIONS))
    return value


@dataclass(frozen=True)
class CudaGraphConfig:
    """Static-shape contract for one batch-1 decode graph."""

    ctx: int
    max_new_tokens: int
    batch: int = 1
    q_len: int = 1

    def __post_init__(self) -> None:
        if self.batch != 1:
            raise ValueError("the EXP-p0-03 gate supports batch=1 only")
        if self.q_len != 1:
            raise ValueError("the EXP-p0-03 gate supports q_len=1 only")
        if self.ctx <= 0 or self.max_new_tokens <= 0:
            raise ValueError("ctx and max_new_tokens must be positive")

    @property
    def max_cache_len(self) -> int:
        return self.ctx + self.max_new_tokens

    def validate_timing(self, timed: int) -> None:
        if timed <= 0:
            raise ValueError("timed must be positive")
        effective_timed = max(timed, 20)
        if effective_timed > self.max_new_tokens:
            raise ValueError(
                "effective timed=%d exceeds max_new_tokens=%d; StaticCache timing horizon would overflow"
                % (effective_timed, self.max_new_tokens)
            )


@dataclass
class GraphBuffers:
    """Fixed-address inputs and token-selection scratch buffers."""

    token: Any
    cache_position: Any
    position_ids: Any
    attention_mask: Any
    next_token: Any
    filled_len: Any = None

    @classmethod
    def allocate(
        cls,
        device: Any,
        *,
        batch: int,
        max_cache_len: int | None = None,
        dtype: Any | None = None,
    ) -> "GraphBuffers":
        import torch

        if batch != 1:
            raise ValueError("GraphBuffers currently supports batch=1 only")
        if max_cache_len is None:
            max_cache_len = 1
        if dtype is None:
            dtype = torch.float32
        return cls(
            token=torch.zeros((batch, 1), dtype=torch.long, device=device),
            cache_position=torch.zeros((1,), dtype=torch.long, device=device),
            position_ids=torch.zeros((batch, 1), dtype=torch.long, device=device),
            attention_mask=torch.zeros((batch, 1, 1, max_cache_len), dtype=dtype, device=device),
            next_token=torch.zeros((batch, 1), dtype=torch.long, device=device),
            # ★ 设备端"已填充长度"：decode attention kernel 的**循环上界**。
            #   形状静态（(1,) int32）、地址固定，每步在图**外** `fill_`。
            #   这就是"launch 形状不随长度变，但答案随长度变"的全部机制：
            #   CUDA Graph 记录的是 launch，kernel 在运行时从设备内存读边界。
            filled_len=torch.zeros((1,), dtype=torch.int32, device=device),
        )

    def set_step(self, *, token: int, position: int) -> None:
        """Mutate graph inputs in place; never replace their tensor objects."""
        self.token.fill_(int(token))
        self.cache_position.fill_(int(position))
        self.position_ids.fill_(int(position))
        if self.filled_len is not None:
            self.filled_len.fill_(int(position) + 1)

    def addresses(self) -> tuple[int, int, int, int, int, int]:
        tensors = (
            self.token,
            self.cache_position,
            self.position_ids,
            self.attention_mask,
            self.next_token,
        )
        if self.filled_len is not None:
            tensors = tensors + (self.filled_len,)
        return tuple(int(t.data_ptr()) for t in tensors)


class CudaGraphCaptureError(RuntimeError):
    """Capture/replay failure with an explicit lifecycle stage."""

    def __init__(self, stage: str, cause: BaseException):
        self.stage = stage
        self.cause = cause
        super().__init__("CUDA Graph %s failed: %s: %s" % (stage, type(cause).__name__, cause))


class CudaGraphCorrectnessError(CudaGraphCaptureError):
    """Graph replay completed but failed the eager-reference correctness gate."""

    def __init__(self, report: dict[str, Any]):
        self.report = report
        cause = RuntimeError(
            "graph/eager correctness gate failed: KL=%s top1=%s"
            % (report.get("kl"), report.get("top1_agreement"))
        )
        super().__init__("correctness", cause)


def _short_error(exc: BaseException, limit: int = 1000) -> str:
    return str(exc).replace("\x00", " ").strip()[:limit]


def failure_record(exc: BaseException) -> dict[str, str]:
    return {
        "stage": getattr(exc, "stage", "unknown"),
        "type": type(getattr(exc, "cause", exc)).__name__,
        "message": _short_error(getattr(exc, "cause", exc)),
    }


def blocked_result(
    *,
    exp_id: str,
    config: dict[str, Any],
    failure: dict[str, str],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a schema-valid blocked result with no performance measurements."""
    payload = empty_result(exp_id, status="blocked")
    # ★ `bit_width` / `group_size` / `dequant` 必须来自 config —— 硬编码 "fp16" 会让
    #   **被阻塞的量化格**以 fp16 的名义落盘（P3 的第二处同类坑；第一处是
    #   `build_payload_cudagraph` 的 config 段）。实测过：w8a16 × cudagraph 的一次
    #   capture OOM 写出的 payload 里 bit_width=fp16，那是错的记录。
    bit_width = str(config.get("bit_width") or "fp16")
    payload["hypothesis"] = {
        "statement": (
            ("%s OLMoE decode" % bit_width)
            + " can use a continuous StaticCache and capture a fixed-shape single-token CUDA Graph"
        ),
        "refutation_signal": (
            "StaticCache setup, CUDA Graph capture, replay, or correctness comparison fails; "
            "failure must identify its lifecycle stage"
        ),
    }
    payload["config"].update(
        {
            "bit_width": bit_width,
            "group_size": config.get("group_size") if bit_width != "fp16" else None,
            "dequant": (config.get("dequant") or "none") if bit_width != "fp16" else "none",
            "fusion": config.get("fusion") or "none",
            "execution": "cudagraph",
            "kv_cache_dtype": "fp16",
            "ctx": config.get("ctx"),
            "batch": config.get("batch", 1),
            "max_new_tokens": config.get("max_new_tokens"),
            "warmup": config.get("warmup"),
            "timed": config.get("timed"),
        }
    )
    payload["correctness"] = {
        "kl": None,
        "top1_agreement": None,
        "ppl_fp16": None,
        "ppl_this": None,
        "ppl_delta": None,
        "gate": "FAIL",
        "thresholds": {
            "kl_max_fp16_graph": 1e-5,
            "top1_min_fp16_graph": 1.0,
            "note": "blocked before a graph/eager correctness comparison",
        },
    }
    payload["verdict"] = {
        "verdict": "FAIL",
        "outcome": "inconclusive",
        "notes": "feasibility gate blocked; no performance numbers are reported",
    }
    payload["applicability"] = {
        "applies_to": "not established; capture/replay did not complete",
        "does_not_apply_to": "quantized execution, paged attention, other GPUs",
        "why": "this is a diagnostic feasibility result, not a performance result",
    }
    payload["_cudagraph"] = {
        "capture": {"success": False},
        "static_cache": {
            "max_cache_len": (config.get("ctx") or 0) + (config.get("max_new_tokens") or 0),
            "layout": "continuous",
        },
        "failure": failure,
    }
    for section in ("model", "hardware", "runtime"):
        values = (metadata or {}).get(section) or {}
        for key in payload[section]:
            if key in values:
                payload[section][key] = values[key]
    if metadata and metadata.get("cudagraph"):
        payload["_cudagraph"].update(metadata["cudagraph"])
    return payload


def failed_result(
    *,
    exp_id: str,
    config: dict[str, Any],
    correctness: dict[str, Any],
    note: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a schema-valid correctness failure with no performance numbers."""
    payload = blocked_result(
        exp_id=exp_id,
        config=config,
        failure={"stage": "correctness", "type": "CorrectnessGate", "message": note},
        metadata=metadata,
    )
    payload["status"] = "failed"
    payload["correctness"] = {
        "kl": correctness.get("kl"),
        "top1_agreement": correctness.get("top1_agreement"),
        "ppl_fp16": None,
        "ppl_this": None,
        "ppl_delta": None,
        "gate": "FAIL",
        "thresholds": {
            "kl_max_fp16_graph": 1e-5,
            "top1_min_fp16_graph": 1.0,
            "note": note,
        },
    }
    payload["verdict"] = {
        "verdict": "FAIL",
        "outcome": "refuted",
        "notes": "graph/eager correctness gate failed; no performance numbers are reported",
    }
    payload["_cudagraph"]["correctness"] = correctness
    return payload


def compare_logits(reference: list[Any], candidate: list[Any]) -> dict[str, Any]:
    """Compare per-step logits without requiring bitwise equality."""
    import torch

    if len(reference) != len(candidate) or not reference:
        raise ValueError("reference and candidate must have the same non-empty step count")

    # ★ 必须显式搬到 CPU，不能用 `.to(torch.float32)` 只改 dtype。
    #   两边的来源天然不同设备：eager 参照在 GPU 上，而 graph 侧走
    #   `session.logits_cpu()` 回到 CPU。只改 dtype 会留下两个设备，
    #   `torch.cat` 直接报
    #   "Expected all tensors to be on the same device, but found at least two
    #    devices, cuda:0 and cpu!"。
    #   这个 bug 一直藏着，因为在这之前**没有任何一次 capture 成功过** ——
    #   正确性门从来没有真正跑到过。
    def _flatten(x: Any) -> Any:
        return x.detach().reshape(1, -1).to(device="cpu", dtype=torch.float32)

    ref = torch.cat([_flatten(x) for x in reference], dim=0)
    got = torch.cat([_flatten(x) for x in candidate], dim=0)
    ref_logp = torch.log_softmax(ref, dim=-1)
    got_logp = torch.log_softmax(got, dim=-1)
    kl = (ref_logp.exp() * (ref_logp - got_logp)).sum(dim=-1).mean()
    agreement = (ref.argmax(dim=-1) == got.argmax(dim=-1)).to(torch.float32).mean()
    return {
        "kl": max(float(kl.item()), 0.0),
        "top1_agreement": float(agreement.item()),
    }


def graph_correctness_passed(report: dict[str, Any]) -> bool:
    """Single correctness predicate shared by result construction and gating."""
    kl = report.get("kl")
    return bool(
        kl is not None
        and math.isfinite(float(kl))
        and float(kl) >= 0.0
        and float(kl) <= 1e-5
        and report.get("top1_agreement") == 1.0
        and report.get("seed_token_match") is True
        and report.get("token_sequence_match") is True
    )


def replay_timed_step(session: Any) -> Any:
    """Run one user-visible autoregressive graph step.

    Unlike a raw ``graph.replay()``, this advances the host-side greedy token
    and cache position. Formal latency measurements must use this path so
    they represent successive decode tokens rather than the same state.
    """
    return session.replay(synchronize=True)


# Number of kernels kept in the attribution table. 25 covers the whole MoE
# expert path plus attention; the tail is thousands of near-identical
# elementwise launches and is summarised instead of listed.
KERNEL_TABLE_LIMIT = 25


def kernel_table(
    rows: list[dict[str, Any]], *, timed: int, busy_ms: float, limit: int = KERNEL_TABLE_LIMIT
) -> list[dict[str, Any]]:
    """Turn profiler rows into the per-kernel attribution table.

    ★ Why this is a pure function: it is the thing that decides *which*
    hypothesis the next GPU hour is spent on (DRAM traffic vs kernel-count
    floor).  Keeping it free of torch means it can be regression-tested on the
    Mac against hand-computed values, so a units mistake (us vs ms, total vs
    per-call) cannot silently pick the wrong optimisation target.

    `rows` are `{name, count, self_us_total}` as reported by the profiler over
    a window containing `timed` graph replays.  `count` is therefore total
    occurrences in the window, not per token.

    Returned fields:
        self_ms_per_token   contribution to per-token time (sums to busy_ms)
        occurrences_per_token
        mean_us_per_occurrence
        share_pct
    """
    table: list[dict[str, Any]] = []
    for row in rows:
        count = int(row.get("count") or 0)
        self_ms = float(row.get("self_us_total") or 0.0) / 1000.0
        if count <= 0 or self_ms <= 0.0:
            continue
        per_token = self_ms / max(timed, 1)
        occ = count / float(max(timed, 1))
        table.append(
            {
                "name": str(row.get("name") or "?"),
                "self_ms_per_token": round(per_token, 5),
                "occurrences_per_token": round(occ, 3),
                "mean_us_per_occurrence": round(per_token / occ * 1000.0, 3) if occ else None,
                "share_pct": round(100.0 * per_token / busy_ms, 2) if busy_ms else None,
            }
        )
    table.sort(key=lambda r: r["self_ms_per_token"], reverse=True)
    return table[:limit]


def _profiler_rows(prof: Any) -> list[dict[str, Any]]:
    """Extract `{name, count, self_us_total}` from a torch profiler, if present."""
    try:
        events = prof.key_averages()
    except Exception:  # noqa: BLE001
        return []
    rows = []
    for event in events:
        self_us = getattr(event, "self_device_time_total", 0.0)
        if self_us and self_us > 0:
            rows.append(
                {
                    "name": getattr(event, "key", "?"),
                    "count": getattr(event, "count", 0),
                    "self_us_total": float(self_us),
                }
            )
    return rows


def measure_graph_gap(
    session: Any, *, warmup: int = 5, timed: int = 20, repeats: int = 3
) -> dict[str, Any]:
    """Measure graph replay wall time and profiler GPU busy time.

    Each repeat starts from the same eager-prefilled state. Resetting happens
    outside the timed window so StaticCache capacity and setup work cannot
    contaminate the decode measurement.
    """
    import time

    torch = session.torch
    warmup = max(warmup, 5)
    timed = max(timed, 20)
    wall_samples: list[float] = []
    event_samples: list[float] = []

    for _ in range(repeats):
        session.reset()
        for _ in range(warmup):
            session.replay(synchronize=False)
        session.stream.synchronize()
        session.reset()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(session.stream)
        t0 = time.perf_counter()
        for _ in range(timed):
            replay_timed_step(session)
        session.stream.synchronize()
        wall_samples.append((time.perf_counter() - t0) * 1000.0 / timed)
        end.record(session.stream)
        session.stream.synchronize()
        event_samples.append(start.elapsed_time(end) / timed)

    session.reset()
    for _ in range(warmup):
        session.replay(synchronize=False)
    session.stream.synchronize()
    session.reset()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(timed):
            replay_timed_step(session)
        session.stream.synchronize()
    events = [event for event in prof.key_averages() if event.self_device_time_total > 0]
    busy_ms = sum(event.self_device_time_total for event in events) / timed / 1000.0
    kernels = sum(event.count for event in events) / float(timed)
    # ★ 为什么这里必须记录 per-kernel 表：
    #   p0-04 留下的 1.59 ms 残差有两个互斥解释 ——
    #     H-A 额外的 DRAM 流量（gather 拷贝）
    #     H-B kernel 数 × 单 kernel 最短执行时间（1,073 个 kernel，均值 4.88 µs）
    #   两者的修法完全不同（grouped GEMM vs 融合/减 kernel）。
    #   在此之前残差只是散文里的一个猜测；这张表把它变成测量。
    table = kernel_table(_profiler_rows(prof), timed=timed, busy_ms=busy_ms)
    wall = summarize(wall_samples)
    event = summarize(event_samples)
    host = max(wall["p50"] - busy_ms, 0.0)
    return {
        "status": "measured",
        "method": "perf_counter wall + CUDA event + profiler self_device_time_total",
        "wall_p50_ms": wall["p50"],
        "wall_p99_ms": wall["p99"],
        "gpu_p50_ms": event["p50"],
        "gpu_p99_ms": event["p99"],
        "gpu_busy_ms": busy_ms,
        "host_p50_ms": host,
        "host_share_pct": round(100.0 * host / wall["p50"], 1) if wall["p50"] else None,
        "kernels_per_call": kernels,
        "top_kernels": table,
        "stats": {"wall": wall, "gpu_event": event},
        "caveats": [
            "GPU busy time is profiler self_device_time_total; graph replay is measured as one captured launch.",
            "Reset/prefill work is outside each timed decode window.",
            "nsys is unavailable in the server image, so no third-party launch cross-check is recorded.",
        ],
    }


class StaticCacheDecodeSession:
    """One-token decode session backed by a captured CUDA Graph.

    Prefill is eager. The graph owns only the fixed-shape decode forward. Token
    selection and the host-side position counter remain outside the graph.
    """

    def __init__(
        self,
        model: Any,
        prompt_ids: Any,
        *,
        ctx: int,
        max_new_tokens: int,
        device: str = "cuda",
    ) -> None:
        import torch
        from transformers import StaticCache

        self._cuda_available = bool(torch.cuda.is_available())
        if prompt_ids.ndim != 2 or prompt_ids.shape != (1, ctx):
            raise ValueError("prompt_ids must have shape [1, ctx]")
        self.torch = torch
        self.model = model.eval()
        self.prompt_ids = prompt_ids
        self.device = torch.device(device)
        self.config = CudaGraphConfig(ctx=ctx, max_new_tokens=max_new_tokens, batch=1)
        self.dtype = next(model.parameters()).dtype
        self.cache = StaticCache(
            config=model.config,
            max_cache_len=self.config.max_cache_len,
        )
        self.buffers = GraphBuffers.allocate(
            self.device,
            batch=1,
            max_cache_len=self.config.max_cache_len,
            dtype=self.dtype,
        )
        # ★ 把设备端长度挂到 cache 对象上：被替换的 attention core 通过
        #   `past_key_values` 拿到它（**不用模块级全局变量**）。挂载是幂等的，
        #   不替换 attention 的路径（HF 自己）完全不受影响。
        self.cache._lemoq_filled_len = self.buffers.filled_len
        self.graph = None
        self.stream = None
        self.logits = None
        self._capture_token_id: int | None = None
        self._capture_position: int | None = None
        self.current_token_id: int | None = None
        self.position = self.config.ctx
        self.capture_info: dict[str, Any] = {
            "max_cache_len": self.config.max_cache_len,
            "q_len": self.config.q_len,
            "batch": self.config.batch,
            "capture_success": False,
            "replay_success": False,
        }
        self.prefill()

    def _clear_cache(self) -> None:
        """Clear backing tensors without replacing their addresses."""
        # Transformers owns the cache-layer layout and reset semantics.  Use
        # its public reset API so future cache-layer implementations remain
        # compatible, while preserving the backing tensor addresses required
        # by CUDA Graph replay.
        self.cache.reset()

    def prefill(self) -> Any:
        """Eagerly fill the static cache and seed the next decode token."""
        torch = self.torch
        self._clear_cache()
        positions = torch.arange(self.config.ctx, device=self.device, dtype=torch.long)
        attention_mask = torch.ones(
            (self.config.batch, self.config.ctx), device=self.device, dtype=torch.long
        )
        with torch.no_grad():
            output = self.model(
                self.prompt_ids,
                attention_mask=attention_mask,
                past_key_values=self.cache,
                use_cache=True,
                cache_position=positions,
                position_ids=positions.unsqueeze(0),
            )
        self.current_token_id = int(output.logits[:, -1:, :].argmax(dim=-1).item())
        self.position = self.config.ctx
        if self._cuda_available:
            torch.cuda.synchronize()
        return output

    @property
    def seed_token_id(self) -> int:
        if self.current_token_id is None:
            raise RuntimeError("prefill must run before reading the seed token")
        return self.current_token_id

    def buffer_addresses(self) -> tuple[int, int, int, int, int]:
        return self.buffers.addresses()

    def decode_eager(self) -> Any:
        if self.current_token_id is None:
            raise RuntimeError("prefill must run before decode")
        if self.position >= self.config.max_cache_len:
            raise RuntimeError("decode position exceeded StaticCache capacity")
        self._set_step_inputs()
        with self.torch.no_grad():
            output = self.model(
                self.buffers.token,
                attention_mask=self.buffers.attention_mask,
                past_key_values=self.cache,
                use_cache=True,
                cache_position=self.buffers.cache_position,
                position_ids=self.buffers.position_ids,
            )
        logits = output.logits.detach()
        self.current_token_id = int(logits[:, -1:, :].argmax(dim=-1).item())
        self.position += 1
        if self._cuda_available:
            self.torch.cuda.synchronize()
        return logits

    def _set_step_inputs(self) -> None:
        if self.current_token_id is None:
            raise RuntimeError("prefill must run before decode")
        torch = self.torch
        min_dtype = torch.finfo(self.dtype).min
        self.buffers.token.fill_(self.current_token_id)
        self.buffers.cache_position.fill_(self.position)
        self.buffers.position_ids.fill_(self.position)
        self.buffers.attention_mask.fill_(min_dtype)
        if self.position >= self.config.max_cache_len:
            raise RuntimeError("decode position exceeded StaticCache capacity")
        self.buffers.attention_mask[..., : self.position + 1].zero_()
        # ★ 长度唯一来源：设备端 int32。它在图**外**更新，图里只有它的地址。
        #   注意 mask 仍然每步更新：kernel 不读 mask，但 **eager 臂需要它**，
        #   而两条臂的 host 侧开销必须相同，否则交错对照量到的是 host 差异。
        #   去掉这两次 fill_ 是后续单独的改动（一次一个变量）。
        self.buffers.filled_len.fill_(self.position + 1)

    def capture(self, *, warmup: int = 5) -> dict[str, Any]:
        """Warm up and capture one fixed-shape decode forward."""
        torch = self.torch
        if self.graph is not None:
            return dict(self.capture_info)
        if warmup < 1:
            raise ValueError("warmup must be positive")
        try:
            # ★ 先复位。capture() 不能被假定为"刚构造完、position 还在起点"：
            #   如果在这之前跑过任何 eager 步（例如为了拿同路径参照），
            #   position 会推进到 max_cache_len，warmup 里的 `_set_step_inputs()`
            #   就会抛 "decode position exceeded StaticCache capacity"。
            #   复位之后 capture 与调用顺序无关，也更可复现。
            self.prefill()
            self.stream = torch.cuda.Stream(device=self.device)
            current = torch.cuda.current_stream(self.device)
            self.stream.wait_stream(current)
            initial_token = self.current_token_id
            initial_position = self.position
            with torch.cuda.stream(self.stream), torch.no_grad():
                for _ in range(warmup):
                    self._set_step_inputs()
                    self.model(
                        self.buffers.token,
                        attention_mask=self.buffers.attention_mask,
                        past_key_values=self.cache,
                        use_cache=True,
                        cache_position=self.buffers.cache_position,
                        position_ids=self.buffers.position_ids,
                    )
            self.stream.synchronize()
            self.prefill()
            self.current_token_id = initial_token
            self.position = initial_position
            self.stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self.stream):
                self._set_step_inputs()
            self.stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self.stream), torch.no_grad():
                output = self.model(
                    self.buffers.token,
                    attention_mask=self.buffers.attention_mask,
                    past_key_values=self.cache,
                    use_cache=True,
                    cache_position=self.buffers.cache_position,
                    position_ids=self.buffers.position_ids,
                )
                self.logits = output.logits
            self.stream.synchronize()
            self.graph = graph
            self.capture_info["capture_success"] = True
            self.capture_info["stream"] = "dedicated CUDA stream"
            return dict(self.capture_info)
        except Exception as exc:  # noqa: BLE001
            raise CudaGraphCaptureError("capture", exc) from exc

    def replay(self, *, synchronize: bool = True) -> Any:
        """Replay one decode step; optionally synchronize and advance state."""
        if self.graph is None or self.stream is None or self.logits is None:
            raise RuntimeError("capture must succeed before replay")
        try:
            with self.torch.cuda.stream(self.stream), self.torch.no_grad():
                self._set_step_inputs()
                self.graph.replay()
            if synchronize:
                self.stream.synchronize()
                self.current_token_id = int(self.logits[:, -1:, :].argmax(dim=-1).item())
                self.position += 1
            self.capture_info["replay_success"] = True
            return self.logits
        except Exception as exc:  # noqa: BLE001
            raise CudaGraphCaptureError("replay", exc) from exc

    def set_position(self, position: int) -> None:
        """Move the decode position **without touching the captured graph**.

        ★ 为什么需要它：K4（换长度）判据的全部内容是"在长度 N 捕获、改成 M 重放"。
        长度在这里就是设备端的 `filled_len`（见 `_set_step_inputs`），改它不需要
        重新捕获 —— 也不需要 host 同步。这是本任务立身之本的可测入口。
        """
        if position < 0 or position >= self.config.max_cache_len:
            raise RuntimeError(
                "position %d 超出 StaticCache 容量 %d" % (position, self.config.max_cache_len)
            )
        self.position = int(position)

    def replay_at_position(self, position: int, *, synchronize: bool = True) -> Any:
        """在指定位置重放**已捕获**的图；返回 graph 写出的 logits。

        与 `replay()` 的唯一区别是位置由调用方给（`replay()` 用 self.position 并
        自增）。`synchronize=False` 时不做 sync、也不推进 token，用于纯计时。
        """
        if self.graph is None or self.stream is None or self.logits is None:
            raise RuntimeError("capture must succeed before replay")
        self.set_position(position)
        try:
            with self.torch.cuda.stream(self.stream), self.torch.no_grad():
                self._set_step_inputs()
                self.graph.replay()
            if synchronize:
                self.stream.synchronize()
                self.current_token_id = int(self.logits[:, -1:, :].argmax(dim=-1).item())
                self.position += 1
            self.capture_info["replay_success"] = True
            return self.logits
        except Exception as exc:  # noqa: BLE001
            raise CudaGraphCaptureError("replay", exc) from exc

    def logits_cpu(self) -> Any:
        """Synchronize and copy the most recent graph logits to CPU."""
        if self.stream is None or self.logits is None:
            raise RuntimeError("no graph output is available")
        self.stream.synchronize()
        return self.logits.detach().to("cpu").clone()

    def next_token_cpu(self) -> int:
        if self.stream is None:
            raise RuntimeError("no graph output is available")
        self.stream.synchronize()
        if self.current_token_id is None:
            raise RuntimeError("no graph output is available")
        return self.current_token_id

    def reset(self) -> None:
        """Restore the prefill state for another timed/correctness sequence."""
        self.prefill()
