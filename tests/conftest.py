"""Shared fixtures: the frozen sample and the clean curve panel (built once per session)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from energycurves.data.stitch import build_curve_panel

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample"


@pytest.fixture(scope="session")
def sample_dir() -> Path:
    return SAMPLE


@pytest.fixture(scope="session")
def futures() -> pd.DataFrame:
    return pd.read_parquet(SAMPLE / "futures_daily.parquet")


@pytest.fixture(scope="session")
def panel(futures: pd.DataFrame) -> pd.DataFrame:
    return build_curve_panel(futures)
