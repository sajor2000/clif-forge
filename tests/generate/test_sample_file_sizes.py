"""Committed sample parquet files must stay under GitHub's soft size limit.

GitHub warns (and may eventually block) files over 50 MB. The ICU sample is
intentionally sized so its heaviest table (``vitals``) stays under that soft
limit; this test makes a silent re-growth a CI failure instead of a push warning.
"""

from __future__ import annotations

from pathlib import Path

import pytest

#: GitHub's recommended maximum file size (soft limit). The hard block is 100 MB.
_GITHUB_SOFT_LIMIT_BYTES = 50 * 1024 * 1024

_SAMPLE_DIRS = (
    Path("sample_dataset"),
    Path("sample_full_hospital"),
    Path("demo_output"),
)


@pytest.mark.parametrize("sample_dir", _SAMPLE_DIRS, ids=lambda p: p.name)
def test_committed_sample_parquets_stay_under_github_soft_limit(sample_dir: Path) -> None:
    if not sample_dir.is_dir():
        pytest.skip(f"{sample_dir} is not present")
    offenders = [
        (p.name, p.stat().st_size)
        for p in sorted(sample_dir.glob("*.parquet"))
        if p.stat().st_size > _GITHUB_SOFT_LIMIT_BYTES
    ]
    assert not offenders, (
        f"{sample_dir} has parquet file(s) over GitHub's 50 MB soft limit: "
        + ", ".join(f"{name} ({size / (1024 * 1024):.1f} MB)" for name, size in offenders)
        + ". Shrink --n-patients when regenerating, or move the heavyweight pack off-repo."
    )
