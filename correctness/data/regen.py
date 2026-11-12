# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rebuild the pinned eval subsets from their Hugging Face parquet files using MANIFEST.json.

Usage: python3 regen.py [DATA_DIR]
Downloads each dataset's parquet at the pinned revision, checks its sha256, writes the selected rows
(plus "_index") as JSON lines, and checks the jsonl sha256. Exits non-zero on any mismatch.
"""
import hashlib
import json
import os
import sys

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    manifest = json.load(open(os.path.join(data_dir, "MANIFEST.json")))
    ok = True
    for name, d in manifest["datasets"].items():
        out = os.path.join(data_dir, d["jsonl"])
        if os.path.isfile(out) and sha256(out) == d["jsonl_sha256"]:
            print(f"regen {name}: present, sha ok")
            continue
        pq_path = hf_hub_download(d["repo"], d["file"], revision=d["revision"], repo_type="dataset")
        if sha256(pq_path) != d["parquet_sha256"]:
            print(f"[FAIL regen {name}: parquet sha256 mismatch")
            ok = False
            continue
        rows = pq.read_table(pq_path).to_pylist()
        with open(out, "w") as f:
            for i in d["indices"]:
                f.write(json.dumps({**rows[i], "_index": i}) + "\n")
        if sha256(out) != d["jsonl_sha256"]:
            print(f"[FAIL regen {name}: jsonl sha256 mismatch")
            ok = False
        else:
            print(f"regen {name}: rebuilt {len(d['indices'])} rows, sha ok")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
