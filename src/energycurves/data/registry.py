"""Catalogue of every public series used by the project.

One entry per series: where it comes from (URL + EIA/FRED source key), what it
measures and in which unit. ``scripts/fetch_data.py`` reads this catalogue to
download and parse the raw files; the SQL layer reads ``dim_commodity``.

Units (EIA conventions):
    CL  WTI crude oil, Cushing ............ USD/bbl,   1,000 bbl per contract
    NG  natural gas, Henry Hub ............ USD/MMBtu, 10,000 MMBtu per contract
    RB  RBOB gasoline, New York Harbor .... USD/gal,   42,000 gal per contract
    HO  No. 2 heating oil / ULSD, NYH ..... USD/gal,   42,000 gal per contract

1 bbl = 42 US gallons, so a USD/gal price times 42 is a USD/bbl price.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

GALLONS_PER_BARREL: Final[float] = 42.0

EIA_PET_FUTURES_URL: Final[str] = "https://www.eia.gov/dnav/pet/xls/PET_PRI_FUT_S1_D.xls"
EIA_NG_FUTURES_URL: Final[str] = "https://www.eia.gov/dnav/ng/xls/NG_PRI_FUT_S1_D.xls"
EIA_PET_SPOT_URL: Final[str] = "https://www.eia.gov/dnav/pet/xls/PET_PRI_SPT_S1_D.xls"
EIA_HIST_URL: Final[str] = "https://www.eia.gov/dnav/{area}/hist_xls/{key}{freq}.xls"
FRED_URL: Final[str] = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={key}"

# Last date for which the EIA published NYMEX futures prices
# ("Futures prices after April 5, 2024, are not available").
EIA_FUTURES_LAST_DATE: Final[str] = "2024-04-05"

# Freeze date of the versioned sample: every non-futures series is cut here.
SAMPLE_AS_OF: Final[str] = "2026-11-04"


@dataclass(frozen=True)
class Commodity:
    """Static description of one NYMEX energy futures contract."""

    code: str
    name: str
    unit: str
    usd_per_bbl_factor: float | None
    point_value: float
    point_unit: str
    expiry_rule: str


COMMODITIES: Final[dict[str, Commodity]] = {
    "CL": Commodity(
        code="CL",
        name="WTI light sweet crude oil (Cushing, OK)",
        unit="USD/bbl",
        usd_per_bbl_factor=1.0,
        point_value=1_000.0,
        point_unit="bbl",
        expiry_rule="3 business days before the 25th of the month preceding delivery "
        "(4 if the 25th is not a business day)",
    ),
    "NG": Commodity(
        code="NG",
        name="Natural gas (Henry Hub, LA)",
        unit="USD/MMBtu",
        usd_per_bbl_factor=None,
        point_value=10_000.0,
        point_unit="MMBtu",
        expiry_rule="3rd last business day of the month preceding delivery",
    ),
    "RB": Commodity(
        code="RB",
        name="RBOB regular gasoline (New York Harbor)",
        unit="USD/gal",
        usd_per_bbl_factor=GALLONS_PER_BARREL,
        point_value=42_000.0,
        point_unit="gal",
        expiry_rule="last business day of the month preceding delivery",
    ),
    "HO": Commodity(
        code="HO",
        name="No. 2 heating oil / ULSD (New York Harbor)",
        unit="USD/gal",
        usd_per_bbl_factor=GALLONS_PER_BARREL,
        point_value=42_000.0,
        point_unit="gal",
        expiry_rule="last business day of the month preceding delivery",
    ),
}

CODES: Final[tuple[str, ...]] = tuple(COMMODITIES)

# EIA source keys of futures contracts 1-4, per commodity.
FUTURES_KEYS: Final[dict[str, tuple[str, str, str, str]]] = {
    "CL": ("RCLC1", "RCLC2", "RCLC3", "RCLC4"),
    "NG": ("RNGC1", "RNGC2", "RNGC3", "RNGC4"),
    "RB": tuple(f"EER_EPMRR_PE{k}_Y35NY_DPG" for k in range(1, 5)),  # type: ignore[dict-item]
    "HO": tuple(f"EER_EPD2F_PE{k}_Y35NY_DPG" for k in range(1, 5)),  # type: ignore[dict-item]
}


@dataclass(frozen=True)
class Series:
    """A single non-futures time series (spot, inventory, rate, FX)."""

    series_id: str
    source_key: str
    url: str
    description: str
    unit: str
    code: str | None = None
    region: str | None = None


SPOT_SERIES: Final[tuple[Series, ...]] = (
    Series("WTI", "RWTC", EIA_PET_SPOT_URL, "WTI spot, Cushing OK, FOB", "USD/bbl", "CL"),
    Series("BRENT", "RBRTE", EIA_PET_SPOT_URL, "Brent spot, Europe, FOB", "USD/bbl", None),
    Series(
        "NYH_GASOLINE",
        "EER_EPMRU_PF4_Y35NY_DPG",
        EIA_PET_SPOT_URL,
        "Conventional regular gasoline spot, New York Harbor",
        "USD/gal",
        "RB",
    ),
    Series(
        "NYH_HEATING_OIL",
        "EER_EPD2F_PF4_Y35NY_DPG",
        EIA_PET_SPOT_URL,
        "No. 2 heating oil spot, New York Harbor",
        "USD/gal",
        "HO",
    ),
    Series(
        "NYH_ULSD",
        "EER_EPD2DXL0_PF4_Y35NY_DPG",
        EIA_PET_SPOT_URL,
        "Ultra-low-sulfur No. 2 diesel spot, New York Harbor",
        "USD/gal",
        "HO",
    ),
    Series(
        "HENRY_HUB", "RNGWHHD", EIA_NG_FUTURES_URL, "Henry Hub natural gas spot", "USD/MMBtu", "NG"
    ),
)

INVENTORY_SERIES: Final[tuple[Series, ...]] = (
    Series(
        "NG_STORAGE_L48",
        "NW2_EPG0_SWO_R48_BCF",
        EIA_HIST_URL.format(area="ng", key="NW2_EPG0_SWO_R48_BCF", freq="w"),
        "Working gas in underground storage, Lower 48 states",
        "Bcf",
        "NG",
        "L48",
    ),
    Series(
        "CRUDE_STOCKS_US",
        "WCESTUS1",
        EIA_HIST_URL.format(area="pet", key="WCESTUS1", freq="w"),
        "US ending stocks of crude oil excluding SPR",
        "kbbl",
        "CL",
        "US",
    ),
    Series(
        "GASOLINE_STOCKS_US",
        "WGTSTUS1",
        EIA_HIST_URL.format(area="pet", key="WGTSTUS1", freq="w"),
        "US ending stocks of total gasoline",
        "kbbl",
        "RB",
        "US",
    ),
    Series(
        "DISTILLATE_STOCKS_US",
        "WDISTUS1",
        EIA_HIST_URL.format(area="pet", key="WDISTUS1", freq="w"),
        "US ending stocks of distillate fuel oil",
        "kbbl",
        "HO",
        "US",
    ),
)

RATE_SERIES: Final[tuple[Series, ...]] = (
    Series(
        "DTB3",
        "DTB3",
        FRED_URL.format(key="DTB3"),
        "3-month Treasury bill, secondary market, discount basis",
        "percent",
    ),
)

FX_SERIES: Final[tuple[Series, ...]] = (
    Series(
        "EURUSD",
        "DEXUSEU",
        FRED_URL.format(key="DEXUSEU"),
        "US dollars to one euro, noon buying rate in New York",
        "USD per EUR",
    ),
)

# Natural-gas strip snapshot (live contracts) used for the storage valuation.
NG_STRIP_SNAPSHOT_DATE: Final[str] = "2026-11-04"
NG_STRIP_FIRST_MONTH: Final[str] = "2026-12"
NG_STRIP_LAST_MONTH: Final[str] = "2028-12"
NYMEX_MONTH_CODES: Final[str] = "FGHJKMNQUVXZ"
