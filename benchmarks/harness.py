"""harness — 读 `configs/experiments.yaml`，展开成具体运行，落 `result.json`。

职责边界（重要）：

    本模块负责**编排**：展开矩阵、按 arch 裁剪实验臂、断点续跑、校验并落盘结果。
    它**不**负责加载模型或跑 kernel —— 那是 Phase C 的 runner 的工作。
    因此在没有 GPU 的机器上，`--dry-run` 可以把整条流水线跑通并产出**合法但全为
    `None`(=未测)** 的 result.json，这正是"上机前把一切准备好"的验证手段。

用法：

    python -m benchmarks.harness --list                      # 看展开出哪些运行
    python -m benchmarks.harness --dry-run --scope minimal_credible
    python -m benchmarks.harness --arch sm_89 --list          # 模拟蹲到 4090

断点续跑：`status == "done"` 的运行会被跳过，除非 `--force`。
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.schema import empty_result, validate_result

DEFAULT_MATRIX = "configs/experiments.yaml"
RESULTS_DIR = "results"


# ---------------------------------------------------------------------------
# 载入与展开
# ---------------------------------------------------------------------------
def _yaml():
    try:
        import yaml  # noqa: PLC0415

        return yaml
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("需要 pyyaml 才能读实验矩阵。装：uv pip install pyyaml（%s）" % exc)


def load_matrix(path: str = DEFAULT_MATRIX) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return _yaml().safe_load(fh)


def _slug(value: Any) -> str:
    s = str(value)
    s = s.replace(" ", "")
    s = re.sub(r"[^0-9A-Za-z_.\-]+", "-", s)
    return s.strip("-")


@dataclass
class Run:
    """One concrete (experiment, factor combination) pair."""

    exp_id: str
    base_id: str
    stage: str
    kind: str
    config: dict[str, Any] = field(default_factory=dict)
    requires: list[str] = field(default_factory=list)
    combined: bool = False
    hypothesis: str | None = None
    refutation_signal: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "exp_id": self.exp_id,
            "base_id": self.base_id,
            "stage": self.stage,
            "kind": self.kind,
            "config": self.config,
            "requires": self.requires,
            "combined": self.combined,
            "hypothesis": self.hypothesis,
            "refutation_signal": self.refutation_signal,
        }


def arms_for_arch(matrix: dict[str, Any], arch: str) -> list[str]:
    return list((matrix.get("arch_arms") or {}).get(arch, []))


def expand(
    matrix: dict[str, Any], *, arch: str | None = None, scope: str | None = None
) -> list[Run]:
    """Cross-product each experiment's `factors` (and `ctx`), applying arch gating.

    Arch gating: an experiment listing `requires: [fp8]` is dropped when the target arch
    does not provide `fp8`. This is how "蹲到 4090 还是 5090" needs no code change.
    """
    available = set(arms_for_arch(matrix, arch)) if arch else None
    include_ids: set[str] | None = None
    if scope:
        scopes = matrix.get("scope") or {}
        if scope not in scopes:
            raise KeyError("未知 scope %r；可用：%s" % (scope, list(scopes)))
        inc = scopes[scope].get("include") or []
        include_ids = None if "*" in inc else set(inc)

    runs: list[Run] = []
    for exp in matrix.get("experiments") or []:
        base_id = exp["id"]
        if include_ids is not None and base_id not in include_ids:
            continue
        requires = list(exp.get("requires") or [])
        if available is not None and not set(requires).issubset(available):
            missing = sorted(set(requires) - available)
            runs.append(
                Run(
                    exp_id=base_id + "__SKIPPED-arch",
                    base_id=base_id,
                    stage=exp.get("stage", ""),
                    kind=exp.get("kind", ""),
                    config={"skipped_because": "arch %s lacks %s" % (arch, missing)},
                    requires=requires,
                    hypothesis=exp.get("hypothesis"),
                    refutation_signal=exp.get("refutation_signal"),
                )
            )
            continue

        factors = exp.get("factors") or {}
        keys = list(factors.keys())
        value_lists = [list(factors[k]) for k in keys]

        ctxs = exp.get("ctx")
        ctx_list = ctxs if isinstance(ctxs, list) else ([ctxs] if ctxs else [None])

        combos = list(itertools.product(*value_lists)) if keys else [()]
        for combo in combos:
            factor_cfg = dict(zip(keys, combo))
            for ctx in ctx_list:
                cfg = dict(factor_cfg)
                if ctx is not None:
                    cfg["ctx"] = ctx
                suffix_bits = ["%s=%s" % (k, _slug(v)) for k, v in factor_cfg.items()]
                if ctx is not None:
                    suffix_bits.append("ctx=%s" % ctx)
                exp_id = base_id if not suffix_bits else "%s__%s" % (base_id, "_".join(suffix_bits))
                runs.append(
                    Run(
                        exp_id=exp_id,
                        base_id=base_id,
                        stage=exp.get("stage", ""),
                        kind=exp.get("kind", ""),
                        config=cfg,
                        requires=requires,
                        combined=bool(exp.get("combined")),
                        hypothesis=exp.get("hypothesis"),
                        refutation_signal=exp.get("refutation_signal"),
                    )
                )
    return runs


# ---------------------------------------------------------------------------
# 结果落盘
# ---------------------------------------------------------------------------
def result_path(exp_id: str, root: str = RESULTS_DIR) -> Path:
    return Path(root) / exp_id / "result.json"


def is_done(exp_id: str, root: str = RESULTS_DIR) -> bool:
    p = result_path(exp_id, root)
    if not p.exists():
        return False
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh).get("status") == "done"
    except Exception:  # noqa: BLE001
        return False


def write_result(payload: dict[str, Any], root: str = RESULTS_DIR) -> tuple[Path, list[str]]:
    """Validate then write. Returns (path, problems). Problems are advisory but recorded."""
    problems = validate_result(payload)
    p = result_path(payload["exp_id"], root)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return p, problems


def build_skeleton(run: Run, *, status: str = "planned") -> dict[str, Any]:
    """A template-valid result carrying the run's hypothesis and config, all metrics None."""
    payload = empty_result(run.exp_id, status=status)
    payload["hypothesis"] = {
        "statement": run.hypothesis or "TODO: 跑之前必须写下假设",
        "refutation_signal": run.refutation_signal or "TODO: 跑之前必须写下反证信号",
    }
    cfg = payload["config"]
    cfg["bit_width"] = run.config.get("bit_width")
    cfg["group_size"] = run.config.get("group_size")
    cfg["dequant"] = run.config.get("dequant")
    cfg["execution"] = run.config.get("execution")
    cfg["ctx"] = run.config.get("ctx")
    payload["_run"] = run.to_dict()
    payload["_note"] = (
        "骨架：所有可测值为 None（=未测，不是 0）。"
        "Phase C 的 runner 填充分布式字段后才可把 status 改为 done。"
    )
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _detect_arch(root: str = RESULTS_DIR) -> str | None:
    probe = Path(root) / "machine_probe.json"
    if not probe.exists():
        return None
    try:
        with open(probe, encoding="utf-8") as fh:
            data = json.load(fh)
        devs = (data.get("torch") or {}).get("devices") or []
        if devs:
            return devs[0].get("compute_capability")
    except Exception:  # noqa: BLE001
        return None
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", default=DEFAULT_MATRIX)
    ap.add_argument(
        "--arch",
        default=None,
        help="目标 compute capability（默认从 results/machine_probe.json 读）",
    )
    ap.add_argument("--scope", default=None, help="minimal_credible | full")
    ap.add_argument("--list", action="store_true", help="只列出展开结果")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="为每个运行写一个合法的骨架 result.json（值全为 None）",
    )
    ap.add_argument("--force", action="store_true", help="即使 status=done 也重写")
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    args = ap.parse_args(argv)

    matrix = None
    try:
        matrix = load_matrix(args.matrix)
    except (RuntimeError, FileNotFoundError) as exc:
        print("错误: %s" % exc, file=sys.stderr)
        return 3
    arch = args.arch or _detect_arch(args.results_dir)
    if arch:
        print("arch = %s  ->  arms: %s" % (arch, arms_for_arch(matrix, arch)))
    else:
        print("arch = <未知>  未找到 results/machine_probe.json；不做 arch 裁剪")

    runs = expand(matrix, arch=arch, scope=args.scope)
    real = [r for r in runs if not r.exp_id.endswith("__SKIPPED-arch")]
    skipped = [r for r in runs if r.exp_id.endswith("__SKIPPED-arch")]
    print("expanded: %d run(s)  (%d skipped by arch gating)" % (len(real), len(skipped)))

    if args.list:
        for r in runs:
            mark = "SKIP " if r.exp_id.endswith("__SKIPPED-arch") else "     "
            print("  %s%s" % (mark, r.exp_id))
        return 0

    if not args.dry_run:
        print(
            "注意：harness 只做编排。真正的测量由 Phase C 的 runner 完成；"
            "本次没有 GPU 运行，未产出任何测量值。",
            file=sys.stderr,
        )

    written = 0
    bad = 0
    for r in real:
        if not args.force and is_done(r.exp_id, args.results_dir):
            print("  skip (done): %s" % r.exp_id)
            continue
        path, problems = write_result(build_skeleton(r), root=args.results_dir)
        written += 1
        if problems:
            bad += 1
            print("  ✗ %s -> %s" % (r.exp_id, "; ".join(problems)), file=sys.stderr)
        else:
            print("  ✓ %s" % path)

    print("\nwrote %d skeleton result(s); %d with schema problems" % (written, bad))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
