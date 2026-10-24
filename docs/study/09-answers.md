# 09｜练习答案与易错点

先独立写，再核对。这里给的是判断路径，不鼓励背数字而不看配置。

## 架构

1. 权重 `[W]` 与 ctx 无关，ctx 增长会增加 KV/attention 工作和总步长，使固定权重流占比变小；量化只作用于其中一部分。
2. RoPE 是默认关候选、独立轮次；p4c-01 v3 是已采纳路径的另一轮矩阵。直接叠加会把不同时间的基线当成一个基线。
3. `runner.run_cudagraph` 设置 `attn=kernel` → `attn_decode.replace_attention_cores` → `_decode_forward_impl` → `attn_decode_forward` → `_attn_triton._attn_split_kernel`（再 `_attn_combine_kernel`）。

## 量化

1. scale/解包有开销，且权重只占一步约 42%；attention、路由和其他算子不会随位宽等比例下降。
2. `QuantLinear` 的 fused 处理 attention 等 dense 线性层；`QuantFusedExperts` 与量化 grouped kernel 处理堆叠后的专家 gate/up/down 权重。
3. 5 窗时 KL 均值 0.0046 过门（top-1 骑线 3/165），10 窗复测被单窗重尾（w7=0.2247）拽到 0.0262 超门——门估计对样本量敏感，PASS 不稳健 ⇒ 已 de-publish（照跑、只作 evidence）。教训：贴线的结论要用预注册的更大样本复核，而不是当成稳定边界。
4. 不是。42% 是测量/归因，约 1.74× 是把权重路径理想化为零耗时后的推导上限。只用四舍五入的 42% 手算得 1.72×；精确份额对应记录的约 1.74×。

**手算提醒**：`round(-63.5)` 采用 ties-to-even 为 `-64`；非整除输入列要补齐以排组，但重建后裁回原宽。W8 每行每组 scale 占 2 B；不要遗漏 scale 费用。

## Grouped MoE

1. 图捕获固定的是 launch 拓扑、grid、地址；设备端 id 只选择已有权重行。当前 token 若改变 K，grid 与中间 buffer 形状也会变，必须重捕获或设计固定容量掩码。本项目 K=8 固定。
2. 把 `top_w` 与 down 的点积放进 fp32 累加，免去 `(K,H)` fp16 中间张量与中途舍入；数值不一定与 gather 逐位相同，要比高精度 oracle。
3. 不等于。3.22 GB 是路径的**解析字节/拷贝税**，额外访问可能命中 L2；记录也专门区分“额外流量”和“额外 DRAM 流量”。

假张量偏移：`e=2,n=3,k=4,F=5,H=8`，gate 是 `2·(2·5·8)+3·8+4=188`，up 是 `2·80+(5+3)·8+4=228`；down `(E,H,F)` 中 `e=2,n=3,f=4` 是 `2·(8·5)+3·5+4=99`。gate `SPLIT_K=2` 时两个分片先写 fp32 部分和；非线性 SiLU 只能在求和后应用，所以需 `act_combine`。

## Dense 与 attention

1. `filled_len` 只在固定地址的设备 scalar 中变，grid 按静态 `NUM_SPLITS,heads` 启动；kernel 内读它决定循环边界。
2. 越界槽位的 qk=0 仍可能贡献 `exp(0−m)`，抬高 softmax 分母并改变输出；须在 max/exp 前设 `−inf`。
3. 8.14→5.15 µs 是某投影 kernel 的每次调用，不覆盖专家、attention、norm、host 等整步费用。

split-KV 手算题：`M=2`，权重为 `1` 与 `exp2(−1)=0.5`；分子 `[1,3]+0.5·[2,4]=[2,5]`，分母 `1+0.5·2=2`，输出 `[1,2.5]`。分别做 `acc/l` 后直接平均是错的，因为各 split 的 log-sum-exp 归一化基准不同。

## Norm 与候选

1. Python API 的一次调用可以展开成 reduce、cast、mul 等多个 GPU kernel；必须看 profiler 的实际 launch。
2. RoPE 分散在 neg、cat、两次 mul 与 add；单个名字只给其中一部分，还要按调用链排除同 family 的其他用途。
3. 不会。开关在构建/捕获前的 Python 相位决定 forward，graph replay 只重放已捕获的 GPU 操作；须重新构建并捕获。

RMSNorm 小题：`mean(x²)=(9+16)/2=12.5`，`r=1/√12.5`，未舍入的输出为 `[6/√12.5,4/√12.5]`。真正 fp16 路径要按实现规定的位置舍入。RoPE 小题的半切取负得到 `[-3,-4,1,2]`。

## CUDA Graph

1. host 调用一次 `graph.replay()`，GPU 仍执行捕获时录下的各 kernel；图减少的是逐次 launch 的 host 间隔。融合 norm 后可在另一轮图里看见 539 kernel/步。
2. `filled_len` 可在固定地址设备 scalar 中改；`q_len` 变了会改变输入/中间形状和 launch 参数，当前图不能复用。
3. prefill 与权重量化都在 capture 前图外；正常 replay 后的 greedy `argmax().item()` 和 host position 递增也在图外。

CPU 缓冲练习应输出 `same addresses: True`、`token/position/filled: 7 3 4`。重新赋值张量对象会换地址；`fill_` 不换。

## Profiling

1. event 是 stream 区间，busy 是 profiler 各 kernel self device time 的和，wall 是 host 可见计时块；测量工具与同步/图外工作的边界不同，不能互换。
2. eager 模块级解释 graph 的链条失效；graph 本身的 kernel 名/次数/self time 表仍是直接测量，可做 kernel 级分类。
3. 解析字节和冷态计时可估带宽达成率，但寄存器 stall 需要相应硬件计数器。解析 occupancy 上界也不是 stall 证据。
4. `+6%>5%`，整轮 VOID；token 相同只通过正确性条件，不能救性能轮次。

RoPE 有效轮的 off 平均 `(1.9302+2.0009)/2≈1.9655`、on 平均 `(1.7813+1.8200)/2≈1.8007`，相对 off 快 `(1.9655−1.8007)/1.9655≈8.39%`。第一个 off/on 对单独相除不能代表预注册的同轮均值口径。

## 模拟面试

1. 必须画出 `gate_up(E,2F,H)`、`down(E,H,F)`、`gate grid=(K,ceil(F/BN),split)`、`down grid=(ceil(H/BN),split)`；id 进设备权重指针，不进 grid；指出 `SPLIT_K>1` 的 partial/reduce。
2. 写 `s=absmax/127`、`q=round(w/s)`、`ŵ=q·s`、code+scale 字节；用权重仅占 42% 解释端到端上限，别说“8-bit 所以 2×”。
3. prefill 初始化固定 cache、warm/JIT 在 capture 前、capture 一步、replay 改固定地址内容；HF `where` 使 host 可见输出形状依赖数据，`filled_len` 仅控制设备循环。
4. 不能。首尾金丝雀 `+6%`，整轮 VOID；即使 token 一致、treatment 快，也不能发表速度结论。
5. 用 L2 flush 后解析字节/冷计时/实测带宽说“达成率”，同时明确不能推断 stall 原因；换允许 ncu 计数器的机器再测寄存器/访存 stall 与 achieved occupancy。
