"""Free-text device catalogs for DDL columns with no consortium mCIDE.

``vent_brand_name`` uses real Hamilton Medical ICU/transport ventilator model
names. ``dialysis_machine_name`` uses common CRRT platform labels. Neither list
is claimed as mCIDE.
"""

from __future__ import annotations

__all__ = ["DIALYSIS_MACHINE_NAMES", "VENT_BRAND_NAMES", "pick_catalog"]

#: Hamilton Medical mechanical ventilators (ICU / emergency / transport).
VENT_BRAND_NAMES: tuple[str, ...] = (
    "HAMILTON-C6",
    "HAMILTON-G5",
    "HAMILTON-C3",
    "HAMILTON-C1",
    "HAMILTON-T1",
)

#: Common CRRT platform labels.
DIALYSIS_MACHINE_NAMES: tuple[str, ...] = (
    "Prismaflex",
    "PrisMax",
    "NxStage System One",
    "multiFiltratePRO",
)


def pick_catalog(names: tuple[str, ...], key: str) -> str:
    """Deterministic catalog pick from a stable string key (device/stay id)."""
    return names[sum(ord(c) for c in key) % len(names)]
