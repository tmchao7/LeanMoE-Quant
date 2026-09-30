#!/usr/bin/env python3
"""check_docs — 校验仓库内 Markdown 的**内部链接**与**表格列数**。

为什么需要它：本仓库在一次文档整理中，`PLAN.md` 的跨仓库链接多写了一层 `../`、
`docs/*.md` 少写了一层，导致**每一条外部项目链接都是死链**而没人发现。这个检查把
那类错误变成回归测试。

设计约束：
    - 只用标准库（Mac 上默认 `python3` 是 3.9.6，且不能依赖第三方包）；
    - 跳过代码块内部的 `|` 与 `](`（fenced code 是字面量）；
    - 把 `[T](ctx)` 这类"伪链接"排除：目标是已知站点/文件才算链接，否则跳过并计数。

用法：
    python3 -m tools.check_docs            # 检查整个仓库
    python3 -m tools.check_docs --quiet
"""

from __future__ import annotations

import argparse
import os
import re
import sys

# 允许的"非文件"链接目标前缀
_SCHEME_RE = re.compile(r"^(?:[a-zA-Z][a-zA-Z0-9+.\-]*:|//)")
# 链接形如 ](target) 或 ](<target>)
_LINK_RE = re.compile(r"\]\(\s*<([^>]+)>\s*\)|\]\(\s*([^)\s]+)\s*\)")


def iter_markdown_files(root: str) -> list[str]:
    """Return repo-relative paths of tracked-looking markdown files.

    Deliberately skips heavy/irrelevant trees so the check is fast and stable.
    """
    skip_dirs = {
        ".git",
        ".venv",
        ".venv-mac",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "build",
        "dist",
        "results",
        "models",
        "hf_cache",
    }
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for name in filenames:
            if name.endswith(".md"):
                full = os.path.join(dirpath, name)
                out.append(os.path.relpath(full, root))
    return sorted(out)


def _split_lines_outside_fences(text: str):
    """Yield (line_number, line, in_fence) for each line."""
    in_fence = False
    for i, line in enumerate(text.split("\n"), start=1):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            yield i, line, True
            continue
        yield i, line, in_fence


def check_links(root: str, files: list[str]) -> list[str]:
    """Return a list of human-readable problems with internal links."""
    problems: list[str] = []
    for rel in files:
        path = os.path.join(root, rel)
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        except OSError as exc:
            problems.append("%s: 无法读取 (%s)" % (rel, exc))
            continue

        base = os.path.dirname(path)
        for lineno, line, in_fence in _split_lines_outside_fences(text):
            if in_fence:
                continue
            for m in _LINK_RE.finditer(line):
                target = m.group(1) or m.group(2) or ""
                if not target or _SCHEME_RE.match(target):
                    continue
                # 去掉锚点与查询串
                path_part = target.split("#", 1)[0].split("?", 1)[0]
                if not path_part:
                    continue
                # 只校验看起来是路径的目标；`[T](ctx)` 这类伪链接跳过
                if not (path_part.endswith(".md") or path_part.startswith(".") or "/" in path_part):
                    continue
                resolved = os.path.normpath(os.path.join(base, path_part))
                if not os.path.exists(resolved):
                    problems.append("%s:%d: 死链 -> %s" % (rel, lineno, target))
    return problems


def check_tables(root: str, files: list[str]) -> list[str]:
    """Return problems where a Markdown table row's column count differs."""
    problems: list[str] = []
    for rel in files:
        path = os.path.join(root, rel)
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        except OSError:
            continue

        block: list[tuple[int, str]] = []

        def flush(block: list[tuple[int, str]]) -> None:
            if len(block) < 2:
                return
            header_pipes = block[0][1].count("|")
            for lineno, row in block[1:]:
                if row.count("|") != header_pipes:
                    problems.append(
                        "%s:%d: 表格列数不一致 (%d vs 表头 %d): %s"
                        % (rel, lineno, row.count("|"), header_pipes, row.strip()[:70])
                    )

        for lineno, line, in_fence in _split_lines_outside_fences(text):
            if not in_fence and line.lstrip().startswith("|"):
                block.append((lineno, line))
            else:
                flush(block)
                block = []
        flush(block)
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="repository root (default: cwd)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    root = os.path.abspath(args.root)
    files = iter_markdown_files(root)
    problems = check_links(root, files) + check_tables(root, files)

    if not args.quiet:
        print("checked %d markdown files" % len(files))
    if problems:
        for p in problems:
            print("  " + p, file=sys.stderr)
        print("FAIL: %d problem(s)" % len(problems), file=sys.stderr)
        return 1
    if not args.quiet:
        print("OK: all internal links resolve and tables are well-formed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
