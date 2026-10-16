# 可选｜将来有 RTX 5090 时的真机练习

这些是**学习演练**，不是新实验 cell；不覆盖现有 `result.json`，不改矩阵轴或 gate。若租到的机器、驱动或模型路径与记录不同，输出只用于个人理解，不并入已有结论。上机前读 [SERVER_RUNBOOK](../reference/SERVER_RUNBOOK.md) 和 [AGENTS](../../AGENTS.md) 的共租协议；先看 `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`，外来进程只记录和等待，不终止。

## 1. Kernel parity

在已配置好的服务器 Python 环境执行现有 GPU 测试：

```bash
/root/miniconda3/bin/python -m pytest -m gpu -q \
  tests/test_kernel_parity.py tests/test_attn_decode.py tests/test_rmsnorm_parity.py \
  tests/test_fused_rope.py tests/test_routing_fusion.py
```

看失败究竟是 shape/dtype 拒绝、数值超 1e-3 K1，还是缺 CUDA/Triton；**不要放宽门**。本条只检验 kernel 与 oracle/HF 的 parity，不产生模型级性能结论。

## 2. 独立图内 profile

模型快照按当时机器的真实本地路径填写；输出放 `/tmp/lemoq-study/`，明确指定 `--json`，防止工具默认写 `results/graph_profile.json`：

```bash
mkdir -p /tmp/lemoq-study
/root/miniconda3/bin/python -m tools.profile_graph \
  --model /root/autodl-tmp/ms/models/allenai--OLMoE-1B-7B-0924/snapshots/master \
  --ctx 1024 --bit-width w8a16 --dequant fused --attn kernel \
  --json /tmp/lemoq-study/graph_profile.json
```

检查输出 `status`、wall/busy/host share、top kernel 次数；把与历史 `results/graph_profile.json` 的差异当作环境/版本诊断，**不做跨轮加速比**。若需要 RoPE 候选 A/B，额外传 `--ab-fused-rope --json /tmp/lemoq-study/rope_ab.json`；必须核对 `tokens_equal`、canary 和每个 run 的 host share，VOID 就保留 JSON 但不报速度。

## 3. 图捕获练习与退出条件

在 [`benchmarks/cudagraph.py`](../../benchmarks/cudagraph.py) 中逐步观察 `prefill()`、`capture()`、`replay_at_position()`；用已有 GPU 测试验证换位置后输出变化。若 capture 失败，看 `CudaGraphCaptureError.stage`，区分 prefill、capture、replay 与正确性问题。结束后退出 Python 进程释放显存。

**不执行** `benchmarks.runner --force`，不运行已完成的 p4b/p4c cell；若以后要正式重测，须按 [AGENTS §6](../../AGENTS.md) 先取得已有结果覆盖许可、归档旧 payload，再按完整协议另开一轮。
