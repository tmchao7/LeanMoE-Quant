#!/usr/bin/env bash
# run_gpu_suite.sh — 上机执行套件（按顺序：门禁 → PPL 写入 → 任务级精度 → batch 扫描 → occupancy）
#
# ★ 设计原则：
#   1. `--dry-run` **打印完整计划**（命令 / 预计时长 / 显存需求 / 验收阈值 / 失败回滚），
#      不执行任何测量。离线（Mac）上只能跑这一个模式。
#   2. 每一步独立可选（`--only <step>`、`--skip <step>`），失败即停（`--no-fail-fast` 可关）。
#   3. 任何一步没跑，对应结果一律留 `planned` / `null`，**不写数字**（硬规则 1）。
#   4. batch 扫描**单列表**，不得与历史 batch=1 数字混表或相除（硬规则 3）。
#
# 用法：
#   ./scripts/run_gpu_suite.sh --dry-run
#   ./scripts/run_gpu_suite.sh --model <MODEL_PATH> --data-dir <DIR>
#   ./scripts/run_gpu_suite.sh --model <M> --only batch_sweep
#   ./scripts/run_gpu_suite.sh --model <M> --gate-cells "EXP-x,EXP-y"
#
set -uo pipefail

MODEL=""
DATA_DIR=""
GATE_CELLS="EXP-p4b-01-truncation__bit_width=w8a16_execution=cudagraph_ctx=1024,EXP-p4b-01-truncation__bit_width=w8a16_execution=eager_ctx=1024"
DRY_RUN=0
ONLY=""
SKIP=""
FAIL_FAST=1
PY="python"

usage() {
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --gate-cells) GATE_CELLS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --only) ONLY="$2"; shift 2 ;;
    --skip) SKIP="$2"; shift 2 ;;
    --no-fail-fast) FAIL_FAST=0; shift ;;
    --python) PY="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数: $1" >&2; usage; exit 2 ;;
  esac
done

STEPS=(gate ppl task_accuracy batch_sweep occupancy)

should_run() {
  local s="$1"
  [[ -n "$ONLY" ]] && [[ "$ONLY" != "$s" ]] && return 1
  [[ -n "$SKIP" ]] && [[ "$SKIP" == "$s" ]] && return 1
  return 0
}

# 每一步的元信息：预计时长 / 显存 / 验收阈值 / 失败回滚
meta() {
  case "$1" in
    gate)
      echo "预计时长: ~10 min/格 | 显存: ~14 GiB（fp16 权重 13.4 GB + KV）"
      echo "验收: 每格 schema 校验 0 problem；gate FAIL 的格子 latency 段全 null"
      echo "回滚: 覆盖前先归档到 results/<EXP>-run/v1_archive/，用 --force 重跑"
      ;;
    ppl)
      echo "预计时长: ~5 min/臂 | 显存: ~14 GiB"
      echo "验收: ppl_fp16/ppl_this/ppl_delta 至少有一个非 null，且 ppl_note 有值"
      echo "回滚: 只写 ppl_* 三个字段；写坏就从归档恢复该 result.json"
      ;;
    task_accuracy)
      echo "预计时长: ~30 min/臂（200 题 MMLU + 100 题 GSM8K） | 显存: ~14 GiB"
      echo "验收: 两种口径（端到端 A + teacher forcing B）都报；标注是子集；"
      echo "      下降 < 2 个点（自定建议值）；FAIL 时不得发性能结论"
      echo "回滚: 结果独立落 results/task_accuracy.json，不影响任何 cell"
      ;;
    batch_sweep)
      echo "预计时长: ~20 min（4 个 batch + 金丝雀） | 显存: batch=8 时可能 >20 GiB（先探）"
      echo "验收: 金丝雀漂移 ≤5%（否则整轮 VOID）；每个 batch 有 n 与 stddev；"
      echo "      ★ 曲线单列表，禁止与历史 batch=1 数字混表或相除"
      echo "回滚: 结果独立落 results/batch_sweep.json；OOM 就降到 batch ∈ {1,2,4}"
      ;;
    occupancy)
      echo "预计时长: ~10 min | 显存: ~14 GiB"
      echo "验收: 每项带 kind=estimate；拿不到 kernel 句柄就保持 planned（不编数字）"
      echo "回滚: 结果独立落 results/occupancy.json"
      ;;
  esac
}

cmd_for() {
  case "$1" in
    gate) printf '%s\n' "$PY -m benchmarks.runner --exp <每格逐一> --model $MODEL --force  [覆盖前自动归档]" "  gate-cells: $(echo "$GATE_CELLS" | tr ',' '\n' | sed 's/^/    - /')" ;;
    ppl) echo "$PY -m tools.quant_ppl --model $MODEL --ctx 1024 --json results/quant_ppl.json" ;;
    task_accuracy) echo "$PY -m tools.task_accuracy --model $MODEL --data-dir $DATA_DIR --json results/task_accuracy.json" ;;
    batch_sweep) echo "$PY -m tools.batch_sweep --model $MODEL --ctx 1024 --json results/batch_sweep.json" ;;
    occupancy) echo "$PY -m tools.occupancy_probe --json results/occupancy.json" ;;
  esac
}

echo "=============================================================="
echo " LeanMoE-Quant 上机套件"
echo " 分支: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
echo " commit: $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
echo " 模型: ${MODEL:-<未指定>}"
echo " 数据: ${DATA_DIR:-<未指定>}"
echo "=============================================================="

if [[ $DRY_RUN -eq 1 ]]; then
  echo
  echo "★ DRY-RUN：以下只打印计划，不执行任何测量。"
  echo
  for s in "${STEPS[@]}"; do
    should_run "$s" || continue
    echo "--- step: $s ---"
    echo "  命令: $(cmd_for "$s")"
    meta "$s" | sed 's/^/  /'
    echo
  done
  echo "=============================================================="
  echo " 上机前的硬约束（写在这里是为了让每个上机的人先看到）："
  echo "   1. 先 nvidia-smi 看有没有同租户；有就等，记录不 kill"
  echo "   2. 首尾同格金丝雀；漂移 >5% ⇒ 整轮 VOID，保留日志不发数字"
  echo "   3. 未测一律写 planned / null（硬规则 1）"
  echo "   4. 不得跨轮比较（硬规则 3）：batch 曲线单列表"
  echo "   5. 覆盖已落盘 result.json 之前先归档"
  echo
  echo " 各工具的 --dry-run 也可以单独看（离线可跑）："
  echo "   $PY -m tools.task_accuracy --dry-run"
  echo "   $PY -m tools.batch_sweep --dry-run"
  echo "   $PY -m tools.occupancy_probe --dry-run"
  echo "   $PY -m tools.ppl_status --dry-run"
  echo "=============================================================="
  exit 0
fi

if [[ -z "$MODEL" ]]; then
  echo "--model 必填（或先跑 --dry-run 看计划）。" >&2
  exit 2
fi

for s in "${STEPS[@]}"; do
  should_run "$s" || { echo "[skip] $s"; continue; }
  echo
  echo "=========== step: $s ==========="
  meta "$s"
  echo "命令: $(cmd_for "$s")"
  echo "--------------------------------"

  case "$s" in
    gate)
      # 门禁：默认跑 p4b 的两个 w8a16 格（已过门的那一档）；可用 --gate-cells 覆盖。
      # ★ 覆盖已落盘 result.json 之前**自动归档**（规则 9）—— 上机时最容易忘的一步。
      ARCH="results/EXP-p4b-01-run/pre_suite_$(date +%Y%m%d-%H%M%S)"
      mkdir -p "$ARCH"
      IFS=',' read -r -a _cells <<< "$GATE_CELLS"
      for EXP_ID in "${_cells[@]}"; do
        [ -z "$EXP_ID" ] && continue
        if [ -f "results/$EXP_ID/result.json" ]; then
          cp -r "results/$EXP_ID" "$ARCH/" && echo "  [archived] $EXP_ID -> $ARCH/"
        fi
        echo "  -> $EXP_ID"
        $PY -m benchmarks.runner --exp "$EXP_ID" --model "$MODEL" --force || {
          echo "[FAIL] gate: $EXP_ID" >&2
          [[ $FAIL_FAST -eq 1 ]] && exit 1
        }
      done
      ;;
    *)
      cmd="$(cmd_for "$s")"
      eval "$cmd" || {
        echo "[FAIL] $s" >&2
        [[ $FAIL_FAST -eq 1 ]] && exit 1
      }
      ;;
  esac
  echo "[ok] $s"
done

echo
echo "=============================================================="
echo " 上机套件跑完。收尾："
echo "   $PY -m tools.summarize_results        # 生成三张汇总表"
echo "   $PY -m tools.ppl_status               # 刷新 PPL 声明清单"
echo "   $PY -m benchmarks.report --check      # schema 校验"
echo "   git status                            # 确认没有误改历史工件"
echo "=============================================================="
