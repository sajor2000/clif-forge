"""Dataset generation manifest: provenance + distinctness sidecar.

Every generated dataset (the master or a derivative) writes a ``manifest.json``
beside its parquet files recording the generator version, the resolved variant
spec, the seed, and a per-table row count + content hash. This marks the master,
makes a variant reproducible from its recipe, and makes distinctness auditable —
two datasets from different specs have different content hashes, and identical
spec+seed reproduces identical hashes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl

from clifforge import __version__

__all__ = ["datasets_are_distinct", "read_manifest", "write_manifest"]

MANIFEST_NAME = "manifest.json"


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(out_dir: str | Path, *, spec: dict[str, Any] | str, seed: int) -> dict[str, Any]:
    """Write ``manifest.json`` into ``out_dir`` and return it.

    ``spec`` is the resolved variant spec (a dict) or the string ``"master"``.
    Every ``*.parquet`` in the directory contributes a ``{rows, sha256}`` entry — the
    ``clif_<table>_2.1_<maturity>`` files and the non-CLIF ``_truth`` spine alike, so
    the whole deliverable is hash-auditable.
    """
    out = Path(out_dir)
    tables: dict[str, dict[str, Any]] = {}
    for parquet in sorted(out.glob("*.parquet")):
        rows = pl.scan_parquet(parquet).select(pl.len()).collect().item()
        tables[parquet.stem] = {"rows": int(rows), "sha256": _sha256(parquet)}
    manifest: dict[str, Any] = {
        "generator_version": __version__,
        "spec": spec,
        "seed": seed,
        "tables": tables,
    }
    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def read_manifest(out_dir: str | Path) -> dict[str, Any]:
    """Read a dataset's ``manifest.json``."""
    data: dict[str, Any] = json.loads((Path(out_dir) / MANIFEST_NAME).read_text(encoding="utf-8"))
    return data


def datasets_are_distinct(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """True when two manifests describe datasets with different content.

    Compares per-table content hashes: identical spec+seed reproduce identical
    hashes (not distinct); any content difference (a different spec or seed)
    yields at least one differing hash.
    """
    ha = {t: v["sha256"] for t, v in a["tables"].items()}
    hb = {t: v["sha256"] for t, v in b["tables"].items()}
    return ha != hb
