"""BTC real-value and index-based price tracker.

This script fetches historical daily and monthly series from the FRED API,
aligns them to a daily frequency, and computes relative indices that compare
BTC purchasing power against commodities and CPI. Outputs include a cleaned CSV
and illustrative plots. A FRED API key is required via the ``FRED_API_KEY``
environment variable.
"""
from __future__ import annotations

import os
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import pandas as pd
import requests

# Configurable parameters
START_DATE = "2017-01-01"
END_DATE = None  # None = up to latest available
BASE_DATE = "2021-11-10"  # Approximate BTC 2021 cycle top

FRED_API_KEY_ENV = "FRED_API_KEY"
FRED_FALLBACK_CSV_ENV = "FRED_FALLBACK_CSV"

FRED_SERIES_MAP: Dict[str, str] = {
    "btc_usd": "CBBTCUSD",
    "oil_usd": "DCOILWTICO",
    "grain_usd": "PWHEAMTUSDM",
    "gold_usd": "GOLDAMGBD228NLBM",
    "elec_index": "CUUR0000SEHF01",
    "water_index": "CUUR0000SEHG",
    "cpi": "CPIAUCSL",
}


SERIES_COLUMNS = [
    "btc_usd",
    "oil_usd",
    "grain_usd",
    "gold_usd",
    "elec_index",
    "water_index",
    "cpi",
]


def fetch_fred_series(
    series_id: str,
    start_date: str,
    end_date: str | None = None,
    *,
    api_key: str | None = None,
    frequency: str | None = None,
) -> pd.Series:
    """
    Fetch observations for a FRED series as a pandas Series with DateTimeIndex.

    The ``FRED_API_KEY`` environment variable must be set. Values of '.' or
    missing entries are converted to NaN. Returned series is named using the
    provided ``series_id``. Passing an ``api_key`` explicitly allows callers to
    avoid repeated environment lookups when fetching multiple series. The
    optional ``frequency`` parameter mirrors the FRED API argument for users who
    need to override the native series frequency.
    """

    api_key = api_key or os.getenv(FRED_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"Environment variable {FRED_API_KEY_ENV} is required to call the FRED API."
        )

    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
    }
    if frequency:
        params["frequency"] = frequency
    if end_date:
        params["observation_end"] = end_date

    response = requests.get(
        "https://api.stlouisfed.org/fred/series/observations", params=params, timeout=30
    )
    response.raise_for_status()

    data = response.json()
    observations = data.get("observations", [])

    dates: list[pd.Timestamp] = []
    values: list[float] = []
    for obs in observations:
        dates.append(pd.to_datetime(obs.get("date")))
        raw_val = obs.get("value", "")
        try:
            values.append(float(raw_val))
        except (TypeError, ValueError):
            values.append(float("nan"))

    series = pd.Series(values, index=pd.DatetimeIndex(dates), name=series_id)
    return series


def fetch_data(start_date: str, end_date: str | None) -> pd.DataFrame:
    """
    Fetch all required time series from the FRED API and return a combined
    DataFrame indexed by date with canonical column names.

    If no API key is present, the function attempts to load a pre-fetched CSV
    from the ``FRED_FALLBACK_CSV`` environment variable to support offline
    execution while keeping the same downstream processing pipeline.
    """

    fetched: dict[str, pd.Series] = {}

    # Fetch API key once up front for clearer error messaging and to avoid
    # repeated environment lookups inside the per-series loop.
    api_key = os.getenv(FRED_API_KEY_ENV)
    fallback_csv = os.getenv(FRED_FALLBACK_CSV_ENV)

    if not api_key:
        if fallback_csv and os.path.exists(fallback_csv):
            return _load_fallback_csv(fallback_csv, start_date, end_date)
        raise RuntimeError(
            "FRED_API_KEY environment variable is not set. Set it to fetch data from FRED "
            "or provide a FRED_FALLBACK_CSV path containing pre-fetched series."
        )

    for col_name, fred_id in FRED_SERIES_MAP.items():
        series = fetch_fred_series(
            fred_id, start_date=start_date, end_date=end_date, api_key=api_key
        )
        fetched[col_name] = series.rename(col_name)

    df = pd.concat(fetched.values(), axis=1)
    df.index.name = "date"
    return df


def _load_fallback_csv(
    path: str, start_date: str, end_date: str | None
) -> pd.DataFrame:
    """
    Load a pre-fetched CSV containing the canonical columns.

    The CSV is expected to have a ``date`` column and at least the columns in
    ``SERIES_COLUMNS``. This path is surfaced via the ``FRED_FALLBACK_CSV``
    environment variable so the script can run in offline environments while
    preserving the downstream processing logic.
    """

    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    missing = [col for col in SERIES_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "Fallback CSV is missing required columns: " + ", ".join(sorted(missing))
        )

    # Restrict the date range here to mirror the API call behaviour for
    # consistency across online/offline data sources.
    df = df.sort_index()
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date) if end_date else None
    if end is not None:
        df = df.loc[(df.index >= start) & (df.index <= end)]
    else:
        df = df.loc[df.index >= start]
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort, restrict to [START_DATE, END_DATE], and resample to daily frequency.

    Any missing values are forward-filled to ensure alignment across series,
    and rows where all tracked series are NaN are dropped.
    """

    df = df.sort_index()
    start = pd.to_datetime(START_DATE)
    end = pd.to_datetime(END_DATE) if END_DATE else None

    if end:
        df = df.loc[(df.index >= start) & (df.index <= end)]
    else:
        df = df.loc[df.index >= start]

    # Resample to daily and forward-fill to align monthly/irregular series.
    df = df.resample("D").ffill()

    # Drop rows where everything is NaN to avoid divide-by-zero issues later.
    df = df.dropna(how="all", subset=SERIES_COLUMNS)
    return df


def find_base_date(df: pd.DataFrame, base_date_str: str) -> pd.Timestamp:
    """
    Return the index label in df closest to base_date_str.
    """

    base_target = pd.to_datetime(base_date_str)
    # ``get_indexer`` with method="nearest" yields position of closest date.
    loc = df.index.get_indexer([base_target], method="nearest")[0]
    return df.index[loc]


def _compute_relative_series(df: pd.DataFrame, columns: Iterable[str], base_date: pd.Timestamp) -> pd.DataFrame:
    """Create *_rel series that equal 1.0 on the base_date."""

    for col in columns:
        base_value = df.loc[base_date, col]
        if pd.isna(base_value) or base_value == 0:
            raise ValueError(f"Base value for {col} is invalid at {base_date}.")
        df[f"{col}_rel"] = df[col] / base_value
    return df


def compute_indices(df: pd.DataFrame, base_date: pd.Timestamp) -> pd.DataFrame:
    """
    Compute relative price indices and BTC purchasing power metrics.

    Formulas follow the project spec so each *_rel equals 1 at the base date,
    BTC_vs_* indices measure BTC's purchasing power against each commodity, and
    real-price series express BTC in constant 2021 dollars (CPI) or in a
    commodity basket.
    """

    df = df.copy()
    df = _compute_relative_series(df, SERIES_COLUMNS, base_date)

    # BTC purchasing power vs each comparator; all dimensionless.
    df["btc_vs_oil"] = df["btc_usd_rel"] / df["oil_usd_rel"]
    df["btc_vs_grain"] = df["btc_usd_rel"] / df["grain_usd_rel"]
    df["btc_vs_gold"] = df["btc_usd_rel"] / df["gold_usd_rel"]
    df["btc_vs_elec"] = df["btc_usd_rel"] / df["elec_index_rel"]
    df["btc_vs_water"] = df["btc_usd_rel"] / df["water_index_rel"]
    df["btc_vs_cpi"] = df["btc_usd_rel"] / df["cpi_rel"]

    # CPI-adjusted BTC price: expresses BTC in constant purchasing power units.
    df["btc_cpi_real_usd"] = df["btc_usd"] * df.loc[base_date, "cpi"] / df["cpi"]

    # Commodity basket weightings (sum to 1.0).
    weights: Dict[str, float] = {
        "oil_usd_rel": 0.30,
        "grain_usd_rel": 0.20,
        "gold_usd_rel": 0.25,
        "elec_index_rel": 0.15,
        "water_index_rel": 0.10,
    }

    basket = sum(df[col] * weight for col, weight in weights.items())
    df["basket_rel"] = basket
    df["btc_vs_basket"] = df["btc_usd_rel"] / df["basket_rel"]

    # Basket-adjusted BTC price in base-date dollars.
    p_btc_0 = df.loc[base_date, "btc_usd"]
    df["btc_basket_real_usd"] = p_btc_0 * df["btc_vs_basket"]

    return df


def plot_indices(df: pd.DataFrame, base_date: pd.Timestamp) -> None:
    """
    Produce two plots:
    1) BTC nominal vs CPI- and basket-adjusted prices (USD terms).
    2) BTC purchasing power vs CPI and vs the commodity basket (base=1 indices).
    """

    plt.style.use("seaborn-v0_8")

    # Plot nominal vs real BTC prices.
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(df.index, df["btc_usd"], label="BTC/USD (nominal)")
    ax1.plot(df.index, df["btc_cpi_real_usd"], label="BTC in 2021 USD (CPI)")
    ax1.plot(
        df.index,
        df["btc_basket_real_usd"],
        label="BTC in 2021 basket USD",
    )
    ax1.axvline(base_date, color="k", linestyle="--", alpha=0.6, label="Base date")
    ax1.set_title("BTC nominal vs CPI- and basket-adjusted prices")
    ax1.set_ylabel("Price (USD)")
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig("plot_btc_nominal_vs_real.png")

    # Plot purchasing power indices.
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(df.index, df["btc_vs_cpi"], label="BTC vs CPI")
    ax2.plot(df.index, df["btc_vs_basket"], label="BTC vs basket")
    ax2.axhline(1.0, color="gray", linestyle=":", label="Index = 1 at base")
    ax2.axvline(base_date, color="k", linestyle="--", alpha=0.6, label="Base date")
    ax2.set_title("BTC purchasing power vs CPI and commodity basket")
    ax2.set_ylabel("Index (base = 1)")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig("plot_btc_vs_cpi_and_basket.png")

    plt.close(fig1)
    plt.close(fig2)


def main() -> None:
    df = fetch_data(START_DATE, END_DATE)
    df = prepare_data(df)

    base_date = find_base_date(df, BASE_DATE)
    p_btc_0 = df.loc[base_date, "btc_usd"]
    print(f"Base date used: {base_date.date()}")
    print(f"BTC price at base date (P_btc_0): {p_btc_0:,.2f} USD")

    df = compute_indices(df, base_date)

    # Save outputs
    df.to_csv("btc_real_value_indices.csv")
    plot_indices(df, base_date)


if __name__ == "__main__":
    main()
