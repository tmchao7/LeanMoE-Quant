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


def test_fetch_models_verifies_byte_count():
    """仓库名写错（少 `-0924`）时必须能立刻发现，而不是安静地下错东西。"""
    with open(os.path.join(SCRIPTS, "fetch_models.sh"), encoding="utf-8") as fh:
        text = fh.read()
    assert "13838721960" in text, "fetch_models.sh 缺少 OLMoE 的字节数校验值"
    assert "allenai/OLMoE-1B-7B-0924" in text
    assert 'allenai/OLMoE-1B-7B"' not in text


def test_scripts_do_not_contain_credentials():
    """脚本里不得出现真实凭据；只允许从 .env 读取。"""
    suspicious = re.compile(r"(api[_-]?key|token|password)\s*=\s*[\"']?[A-Za-z0-9_\-]{16,}", re.I)
    for path in _shell_scripts():
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        for m in suspicious.finditer(text):
            raise AssertionError("%s 里疑似硬编码凭据: %s" % (path, m.group(0)[:40]))
