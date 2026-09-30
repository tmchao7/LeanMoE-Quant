"""test_links — 把"文档死链 / 表格列数不一致"变成回归测试。

背景（真实事故）：本仓库的 `PLAN.md` 跨仓库链接多写了一层 `../`、`docs/*.md` 少写了一层，
导致**每一条** MegaKernel-vLLM / MoE-Flow 链接都是死链而无人发现。这个测试就是那次事故的
回归保护。

只用标准库，因此在 Mac 的 python3.9.6 上也能跑。
"""

from __future__ import annotations

import os

from tools.check_docs import (
    check_links,
    check_tables,
    iter_markdown_files,
    list_external_links,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_no_broken_internal_links():
    files = iter_markdown_files(ROOT)
    problems = check_links(ROOT, files)
    assert problems == [], "死链:\n" + "\n".join(problems)


def test_cross_repo_links_are_not_treated_as_broken(tmp_path):
    """★ 服务器上抓到的 bug：跨仓库链接在独立 clone 里必然解析不到。

    `PLAN.md` 引用 `../MegaKernel-vLLM/README.md` 是**有意为之**（三个项目在磁盘上
    是兄弟目录）。但在服务器 / CI / 评审者 clone 里那个目录不存在。若把它当成死链，
    这条检查在任何独立 checkout 上都会永远失败 —— 那就等于没有检查。
    """
    docs = tmp_path / "docs"
    docs.mkdir()
    # 必须真的走出仓库根：docs/ -> ../.. = tmp_path 之上
    (docs / "a.md").write_text("[x](../../sibling-project/README.md)\n", encoding="utf-8")
    assert check_links(str(tmp_path), ["docs/a.md"]) == []
    # 但必须能被单独列出来，不能静默丢弃
    assert list_external_links(str(tmp_path), ["docs/a.md"])


def test_internal_broken_link_still_fails(tmp_path):
    """仓库**内部**的死链仍然必须失败 —— 放宽跨仓库链接不能顺手把这个也放宽。"""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("[x](missing.md)\n", encoding="utf-8")
    assert len(check_links(str(tmp_path), ["docs/a.md"])) == 1


def test_tables_are_well_formed():
    files = iter_markdown_files(ROOT)
    problems = check_tables(ROOT, files)
    assert problems == [], "表格问题:\n" + "\n".join(problems)


def test_checker_finds_a_deliberately_broken_link(tmp_path):
    """反向测试：校验器必须真的能抓到死链，否则上面的断言是空跑。"""
    md = tmp_path / "a.md"
    md.write_text("[x](nonexistent-file-xyz.md)\n", encoding="utf-8")
    problems = check_links(str(tmp_path), ["a.md"])
    assert len(problems) == 1
    assert "死链" in problems[0]


def test_checker_ignores_external_and_pseudo_links(tmp_path):
    md = tmp_path / "b.md"
    md.write_text(
        "[ext](https://example.com/x.md)\n"
        "[anchor](#section)\n"
        "[math](ctx)\n"
        "[pkg](lemoq.quant)\n",
        encoding="utf-8",
    )
    assert check_links(str(tmp_path), ["b.md"]) == []


def test_checker_finds_ragged_table(tmp_path):
    md = tmp_path / "c.md"
    md.write_text(
        "| a | b |\n" "|---|---|\n" "| 1 | 2 |\n" "| 1 | 2 | 3 |\n",
        encoding="utf-8",
    )
    problems = check_tables(str(tmp_path), ["c.md"])
    assert len(problems) == 1


def test_fenced_code_blocks_are_not_parsed(tmp_path):
    """代码块里的 `|` 与 `](` 是字面量，不能当成表格或链接。"""
    md = tmp_path / "d.md"
    md.write_text(
        "```text\n" "| not | a | table |\n" "[not](a-link.md)\n" "```\n",
        encoding="utf-8",
    )
    assert check_links(str(tmp_path), ["d.md"]) == []
    assert check_tables(str(tmp_path), ["d.md"]) == []


def test_repo_has_no_stale_olmoe_repo_name():
    """`allenai/OLMoE-1B-7B` 不存在；只有带 -0924 的才是真仓库。"""
    offenders = []
    for rel in iter_markdown_files(ROOT):
        if rel.endswith("test_matrix.py"):
            continue
        with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
            text = fh.read()
        for i, line in enumerate(text.split("\n"), 1):
            if "allenai/OLMoE-1B-7B" in line and "allenai/OLMoE-1B-7B-0924" not in line:
                # 允许"不存在"这类说明性文字
                if "不存在" in line or "does not exist" in line:
                    continue
                offenders.append("%s:%d: %s" % (rel, i, line.strip()[:80]))
    assert offenders == [], "残留的错误仓库名:\n" + "\n".join(offenders)
