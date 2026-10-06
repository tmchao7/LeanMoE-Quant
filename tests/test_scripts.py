"""test_scripts — 对 `scripts/` 下 shell 脚本的静态检查。

为什么值得测：这些脚本是唯一会在**计费机器**上跑的东西，而它们的失败模式往往很安静。
尤其是 `bootstrap.sh --no-torch`：它意图是"复用镜像自带的 torch"，但若依赖列表里
仍然带着 `torch==<pin>`，就会把那个已经验证可用的 torch 覆盖掉——真机上很难发现，
因为覆盖之后 `torch.cuda.is_available()` 可能仍然返回 True，只是构建不匹配。
"""

from __future__ import annotations

import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
CONSTRAINTS = os.path.join(ROOT, "configs", "constraints.txt")

# bootstrap.sh 用于在 --no-torch 时剔除 torch 的正则
_TORCH_LINE = re.compile(r"^[ \t]*torch([=<>!~ ]|$)")


def _shell_scripts():
    return sorted(os.path.join(SCRIPTS, f) for f in os.listdir(SCRIPTS) if f.endswith(".sh"))


def test_scripts_directory_is_not_empty():
    assert _shell_scripts(), "scripts/ 下没有 shell 脚本"


def test_all_shell_scripts_parse():
    """`bash -n` 抓语法错误 —— 在计费机器上跑到一半才炸是最贵的失败。"""
    for path in _shell_scripts():
        proc = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
        assert proc.returncode == 0, "%s 语法错误:\n%s" % (path, proc.stderr)


def test_scripts_set_strict_mode():
    """`set -euo pipefail`（或至少 set -u）是这些脚本能安全失败的前提。"""
    for path in _shell_scripts():
        with open(path, encoding="utf-8") as fh:
            head = fh.read(2000)
        assert re.search(r"^set -[a-z]*u", head, re.M), "%s 没有 set -u" % path


def test_no_torch_actually_excludes_torch_from_requirements():
    """★ 回归测试：constraints.txt 里含 torch==<pin>，`--no-torch` 必须把它剔除，
    否则会覆盖镜像自带的、已经验证过能在目标 arch 上跑的 torch。
    """
    assert os.path.exists(CONSTRAINTS), "缺少 configs/constraints.txt"
    with open(CONSTRAINTS, encoding="utf-8") as fh:
        lines = fh.read().split("\n")

    kept = [ln for ln in lines if not _TORCH_LINE.match(ln)]
    excluded = [ln for ln in lines if _TORCH_LINE.match(ln)]

    assert any(
        ln.startswith("torch==") for ln in excluded
    ), "constraints.txt 里没有 torch pin —— 那么这个测试失去了意义，请更新它"
    assert not any(_TORCH_LINE.match(ln) for ln in kept), "剔除后仍残留 torch 行: %s" % [
        ln for ln in kept if _TORCH_LINE.match(ln)
    ]
    # 关键：不能误伤 transformers
    assert any(
        ln.startswith("transformers") for ln in kept
    ), "过滤把 transformers 也剔掉了 —— 正则写错了"


def test_bootstrap_script_uses_the_filter():
    """脚本里必须真的用这个过滤逻辑，而不是只在测试里成立。"""
    with open(os.path.join(SCRIPTS, "bootstrap.sh"), encoding="utf-8") as fh:
        text = fh.read()
    assert "SKIP_TORCH" in text
    assert "grep -vE" in text, "bootstrap.sh 没有对依赖列表做过滤"
    assert "notorch" in text or "no-torch" in text


def test_bootstrap_stops_on_death_test_failure():
    """死亡测试失败必须 `exit` 非零 —— 否则会带着坏环境继续烧钱。"""
    with open(os.path.join(SCRIPTS, "bootstrap.sh"), encoding="utf-8") as fh:
        text = fh.read()
    assert "no kernel image" in text, "bootstrap 没有针对 sm_120 那个经典错误的提示"
    assert re.search(r"exit 4", text), "死亡测试失败后没有退出"


def test_bootstrap_supports_reusing_an_existing_interpreter():
    """云镜像上最有用的路径：复用镜像自带的 torch，而不是重装一个。

    AutoDL 的 "PyTorch 2.8.0 / CUDA 12.8" 镜像里 torch/triton 都已就绪且经平台验证，
    重装只会更慢、并可能引入与镜像 nvcc 不匹配的构建。
    """
    with open(os.path.join(SCRIPTS, "bootstrap.sh"), encoding="utf-8") as fh:
        text = fh.read()
    assert "USE_EXISTING_PY" in text
    assert "--python=*" in text, "缺少 --python=PATH 形式的参数解析"
    assert "EXPLICIT_PY" in text
    # 复用已有解释器时必须先确认它真的有 torch，否则后面所有 GPU 测试都会失败
    assert "import torch" in text, "没有校验目标解释器里是否存在 torch"
    assert "复用已有解释器" in text or "不新建 venv" in text


def test_bootstrap_dry_run_accepts_both_python_forms():
    """`--python` 与 `--python=PATH` 都必须能被解析。

    用 `--no-preflight` 避免真实网络探测 —— 单元测试不该依赖网络，也不该跑几十秒。
    """
    script = os.path.join(SCRIPTS, "bootstrap.sh")
    import sys

    # 复用解释器时必须带 torch，所以显式指向当前测试的 Python（venv）；
    # 裸 `--python` 的情形用 LEMOQ_PYTHON 注入。
    env = dict(os.environ, LEMOQ_PYTHON=sys.executable)
    for args in (
        ["--no-preflight", "--dry-run"],
        ["--no-preflight", "--no-torch", "--dry-run"],
        ["--no-preflight", "--python=%s" % sys.executable, "--dry-run"],
        ["--no-preflight", "--python", "--dry-run"],
    ):
        proc = subprocess.run(
            ["bash", script] + args,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, "%s 失败 (rc=%d):\n%s" % (args, proc.returncode, proc.stderr)
        assert "unknown arg" not in proc.stdout, "%s 未被识别" % args


def test_bootstrap_rejects_a_missing_interpreter(tmp_path):
    """指向不存在的解释器必须干净地失败（退出非零 + 明确报错），而不是继续往下跑。"""
    script = os.path.join(SCRIPTS, "bootstrap.sh")
    proc = subprocess.run(
        [
            "bash",
            script,
            "--no-preflight",
            "--dry-run",
            "--python=%s/definitely-not-a-python" % tmp_path,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode != 0, "不存在的解释器竟然通过了"
    assert "unknown arg" not in proc.stdout


def test_fetch_models_verifies_byte_count():
    """仓库名写错（少 `-0924`）时必须能立刻发现，而不是安静地下错东西。"""
    with open(os.path.join(SCRIPTS, "fetch_models.sh"), encoding="utf-8") as fh:
        text = fh.read()
    assert "13838721960" in text, "fetch_models.sh 缺少 OLMoE 的字节数校验值"
    assert "allenai/OLMoE-1B-7B-0924" in text
    assert 'allenai/OLMoE-1B-7B"' not in text


def test_fetch_models_detects_incomplete_downloads():
    """只统计 *.safetensors 是不够的：下载中断时 `.incomplete` 会留在磁盘上，
    而分片字节数可能"接近但不等于"期望值，容易被当成别的原因去排查。"""
    with open(os.path.join(SCRIPTS, "fetch_models.sh"), encoding="utf-8") as fh:
        text = fh.read()
    assert ".incomplete" in text
    assert "续传" in text or "resume" in text.lower()


def test_fetch_models_supports_both_sources():
    """两个源的实测差距是 7 倍（ModelScope ~15 MiB/s vs hf-mirror ~2.2 MiB/s），
    所以两条路径都必须存在且都指向**可达**的主机。"""
    with open(os.path.join(SCRIPTS, "fetch_models.sh"), encoding="utf-8") as fh:
        text = fh.read()
    assert "modelscope" in text and "snapshot_download" in text
    assert "hf-mirror.com" in text
    # ★ 绝不能把 huggingface.co 当作可用的**下载目标** —— 实测在该环境不可达
    #   （curl 000），即使挂 AutoDL 的学术加速也只有 0.45 MiB/s。
    #   注释里提到它（解释为什么不走）是可以的，所以要排除注释行再判断。
    code = "\n".join(ln for ln in text.split("\n") if not ln.lstrip().startswith("#"))
    assert "huggingface.co" not in code, "代码里出现了 huggingface.co（该环境不可达）"


def test_fetch_models_creates_dirs_before_df():
    """★ 顺序 bug：HF_HOME 默认指向 ~/.cache/huggingface，首次运行时不存在，
    `df` 失败 → 可用空间读成 0 → 误报"磁盘不足"而拒绝下载。"""
    with open(os.path.join(SCRIPTS, "fetch_models.sh"), encoding="utf-8") as fh:
        text = fh.read()
    mkdir_pos = text.find('mkdir -p "${HF_HOME}"')
    df_pos = text.find('df -Pk "${HF_HOME}"')
    assert mkdir_pos != -1, "没有先建目录"
    assert df_pos != -1, "没有磁盘预检"
    assert mkdir_pos < df_pos, "必须在 df 之前建目录，否则首次运行会误判磁盘不足"


def test_fetch_models_check_mode_does_not_need_disk():
    """--check 只做校验、不写盘，不该因为磁盘紧张而被拦住。"""
    with open(os.path.join(SCRIPTS, "fetch_models.sh"), encoding="utf-8") as fh:
        text = fh.read()
    assert 'CHECK_ONLY}" != "1"' in text


def test_scripts_do_not_contain_credentials():
    """脚本里不得出现真实凭据；只允许从 .env 读取。"""
    suspicious = re.compile(r"(api[_-]?key|token|password)\s*=\s*[\"']?[A-Za-z0-9_\-]{16,}", re.I)
    for path in _shell_scripts():
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        for m in suspicious.finditer(text):
            raise AssertionError("%s 里疑似硬编码凭据: %s" % (path, m.group(0)[:40]))


def test_server_gate0_accepts_existing_interpreter_without_a_repo_venv():
    """The rented image uses /root/miniconda3; Gate-0 must not require a new venv."""
    with open(os.path.join(SCRIPTS, "run_server_gate0.sh"), encoding="utf-8") as fh:
        text = fh.read()
    assert "LEMOQ_PYTHON" in text
    assert "/root/miniconda3/bin/python" in text
    assert "找不到可用 Python" in text


# ---------------------------------------------------------------------------
# run_cudagraph_gate.sh —— GPU 机器上的一键门
# ---------------------------------------------------------------------------
GATE = os.path.join(SCRIPTS, "run_cudagraph_gate.sh")


def test_cudagraph_gate_refuses_without_cuda():
    """★ 在 Mac（无 CUDA）上必须**干净地拒绝**，而不是跑出半截结果。

    这条比看起来重要：Mac 上产生的数字绝不能进 results/。如果脚本在无 GPU 的
    机器上"能跑"，就迟早会有人用它写出假结果。
    """
    proc = subprocess.run(
        ["bash", GATE, "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        env=dict(os.environ, LEMOQ_PYTHON=os.path.join(ROOT, ".venv-mac", "bin", "python")),
    )
    assert proc.returncode != 0, "无 CUDA 时竟然返回 0"
    combined = proc.stdout + proc.stderr
    assert "CUDA" in combined


def test_cudagraph_gate_runs_p0_03_before_p0_04():
    """★ 顺序是刻意的：先证明 HF 原版能不能进图，再试固定形状版本。

    反过来的话，p0-04 通过会掩盖"原版到底能不能"这个问题，
    而那个问题决定了我们要不要维护第二套 MoE 实现。
    """
    with open(GATE, encoding="utf-8") as fh:
        text = fh.read()
    i03 = text.find("EXP-p0-03-baseline-cudagraph")
    i04 = text.find("EXP-p0-04-cudagraph-static-moe")
    assert i03 != -1 and i04 != -1, "缺实验 id"
    assert i03 < i04, "p0-03 必须先于 p0-04 执行"


def test_cudagraph_gate_passes_static_moe_only_to_p0_04():
    """`--static-moe` 只能给 p0-04。给了 p0-03 就是换了它的假设。"""
    with open(GATE, encoding="utf-8") as fh:
        text = fh.read()
    assert 'EXP-p0-04-cudagraph-static-moe" "--static-moe"' in text


def test_cudagraph_gate_verifies_model_bytes():
    """加载 13 GB 之前先校验字节数 —— 不完整的模型会让失败看起来像 capture 问题。"""
    with open(GATE, encoding="utf-8") as fh:
        text = fh.read()
    assert "13838721960" in text
    assert ".incomplete" in text


def test_cudagraph_gate_sets_cuda_path_and_hf_home():
    """两个环境变量的坑：HF_HOME 不设会塞满 30 GB 系统盘；
    PATH 不设会让 nvcc/ncu 找不到（它们不在默认 PATH 上）。"""
    with open(GATE, encoding="utf-8") as fh:
        text = fh.read()
    assert "/root/autodl-tmp/hf" in text
    assert "/usr/local/cuda/bin" in text


def test_embedded_python_in_scripts_compiles():
    """★ 脚本里的 heredoc Python 必须语法正确。

    `bash -n` 不会检查 heredoc 里的 Python —— 那是运行时才炸的，
    而"运行时"意味着已经在计费机器上了。
    """
    import re as _re

    pattern = _re.compile(r"<<'PY(?:EOF)?'\n(.*?)\nPY(?:EOF)?\n", _re.S)
    checked = 0
    for name in os.listdir(SCRIPTS):
        if not name.endswith(".sh"):
            continue
        path = os.path.join(SCRIPTS, name)
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        for m in pattern.finditer(text):
            body = m.group(1)
            try:
                compile(body, "%s:heredoc" % name, "exec")
            except SyntaxError as exc:
                raise AssertionError("%s 的嵌入 Python 语法错误: %s" % (name, exc)) from exc
            checked += 1
    assert checked > 0, "没找到任何嵌入的 Python（正则是不是失效了？）"
