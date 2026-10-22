"""Prepare split-isolated caches on CPU from existing WikiText raw text; no download."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch

from tools.w4_calibrate import file_sha256, tokenizer_identity


def prepare(text_dir: Path, out: Path, *, tokenizer, tokenizer_hash: str, revision: str):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("pinned dataset commit SHA required")
    if out.exists():
        raise FileExistsError(out)
    texts = {split: text_dir / (split + ".txt") for split in ("train", "validation", "test")}
    if not all(path.is_file() for path in texts.values()):
        raise ValueError("train.txt, validation.txt and test.txt must already exist")
    raw_hashes = [file_sha256(p) for p in texts.values()]
    if len(set(raw_hashes)) != 3:
        raise ValueError("split text reuse is forbidden")
    out.mkdir(parents=True)
    metadata = {
        "dataset": "wikitext-2-raw-v1",
        "revision": revision,
        "tokenizer_sha256": tokenizer_hash,
        "splits": {},
        "status": "incomplete",
    }
    try:
        for split, path in texts.items():
            # Same nonempty-line joining policy as tools.quant_ppl.
            text = "\n".join(line for line in path.read_text().splitlines() if line.strip())
            ids = (
                tokenizer(text, return_tensors="pt", truncation=False)
                .input_ids.reshape(-1)
                .cpu()
                .long()
            )
            if ids.numel() < 2 or torch.unique(ids).numel() < 2:
                raise ValueError("degenerate text split: " + split)
            target = out / (split + ".pt")
            with target.open("xb") as fh:
                torch.save(ids, fh)
            metadata["splits"][split] = {
                "file_sha256": file_sha256(target),
                "token_sha256": hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
                "source_text_sha256": file_sha256(path),
                "tokens": ids.numel(),
            }
        if len({r["token_sha256"] for r in metadata["splits"].values()}) != 3:
            raise ValueError("cross-split token reuse is forbidden")
        metadata["status"] = "complete"
    finally:
        (out / "data_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="local tokenizer snapshot")
    ap.add_argument("--text-dir", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--out", required=True, help="new CPU cache directory")
    ap.add_argument("--execute", action="store_true", help="prepare CPU caches (no GPU)")
    args = ap.parse_args(argv)
    if not args.execute:
        print(json.dumps({"status": "planned", "device": "cpu", "inputs": vars(args)}))
        return 0
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prepare(
        Path(args.text_dir),
        Path(args.out),
        tokenizer=tokenizer,
        tokenizer_hash=tokenizer_identity(Path(args.model)),
        revision=args.revision,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
