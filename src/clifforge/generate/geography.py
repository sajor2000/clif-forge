"""Synthetic neighbourhood catalog for hospitalization geography ↔ place_based_index.

Fictional ZIP/FIPS/tract codes — not real patient areas. Each neighbourhood carries
a stable deprivation level so ``hospitalization`` geography and in-memory
``place_based_index`` rows for the same ``hospitalization_id`` stay coherent.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

__all__ = ["Neighborhood", "neighborhood_for"]

_FIPS_VERSION = "2020"


@dataclass(frozen=True)
class Neighborhood:
    """One synthetic neighbourhood (all geography codes + latent deprivation)."""

    zipcode_five_digit: str
    zipcode_nine_digit: str
    state_code: str
    county_code: str
    census_tract: str
    census_block_group_code: str
    census_block_code: str
    fips_version: str
    #: Latent deprivation in [0, 1] shared with place_based_index ADI/SVI draws.
    deprivation: float


def _row(
    zip5: str,
    zip9: str,
    state: str,
    county: str,
    tract: str,
    block_group: str,
    block: str,
    deprivation: float,
) -> Neighborhood:
    return Neighborhood(
        zip5,
        zip9,
        state,
        county,
        tract,
        block_group,
        block,
        _FIPS_VERSION,
        deprivation,
    )


#: Finite fictional catalogue. Codes are synthetic and must not be treated as
#: real geographies.
_CATALOG: tuple[Neighborhood, ...] = (
    _row("90001", "900010001", "06", "037", "06037207100", "060372071001", "060372071001001", 0.82),
    _row("90012", "900120123", "06", "037", "06037207300", "060372073002", "060372073002015", 0.71),
    _row("90210", "902100456", "06", "037", "06037262000", "060372620001", "060372620001003", 0.12),
    _row("94102", "941020789", "06", "075", "06075012200", "060750122002", "060750122002008", 0.55),
    _row("94107", "941070234", "06", "075", "06075018000", "060750180001", "060750180001012", 0.38),
    _row("60611", "606110111", "17", "031", "17031080100", "170310801001", "170310801001004", 0.22),
    _row("60637", "606370222", "17", "031", "17031420100", "170314201002", "170314201002019", 0.88),
    _row("60614", "606140333", "17", "031", "17031070100", "170310701003", "170310701003007", 0.18),
    _row("10001", "100010444", "36", "061", "36061009500", "360610095001", "360610095001011", 0.45),
    _row("10027", "100270555", "36", "061", "36061020901", "360610209011", "360610209011022", 0.76),
    _row("11201", "112010666", "36", "047", "36047000100", "360470001002", "360470001002005", 0.33),
    _row("02115", "021150777", "25", "025", "25025010404", "250250104041", "250250104041009", 0.41),
    _row("02139", "021390888", "25", "017", "25017353100", "250173531002", "250173531002014", 0.29),
    _row("19104", "191040999", "42", "101", "42101009100", "421010091001", "421010091001003", 0.67),
    _row("75201", "752011010", "48", "113", "48113000100", "481130001001", "481130001001006", 0.25),
    _row("77002", "770021121", "48", "201", "48201100000", "482011000002", "482011000002018", 0.58),
    _row("30303", "303031232", "13", "121", "13121000100", "131210001001", "131210001001002", 0.49),
    _row("33101", "331011343", "12", "086", "12086000100", "120860001002", "120860001002010", 0.61),
    _row("98101", "981011454", "53", "033", "53033007200", "530330072001", "530330072001007", 0.27),
    _row("80202", "802021565", "08", "031", "08031001701", "080310017011", "080310017011015", 0.35),
)


def neighborhood_for(hospitalization_id: str) -> Neighborhood:
    """Stable neighbourhood for a stay — same id always yields the same row."""
    digest = hashlib.blake2b(hospitalization_id.encode("utf-8"), digest_size=8).digest()
    idx = int.from_bytes(digest, "big") % len(_CATALOG)
    return _CATALOG[idx]
