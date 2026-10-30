"""Small, testable rules shared by the DFW Crash Map dashboard."""

from datetime import date

import pandas as pd


# This is a conservative data-coverage check, not proof that TxDOT's export is exhaustive.
# Confirm the underlying CRIS export dates independently before publishing full-year claims.
YEAR_EDGE_TOLERANCE_DAYS = 7
MAX_UNREPORTED_GAP_DAYS = 14


def compute_year_bounds(data, today=None):
    """Return observed date bounds and years with plausible full-calendar coverage.

    Each selected city must have records near both calendar-year boundaries, records
    in all twelve months, and no observed gap longer than 14 days. Ongoing calendar
    years are excluded. A missing-record day need not mean missing *coverage*, so
    this is a conservative heuristic rather than an official export completeness flag.
    """
    today = today or date.today()
    dated = data.dropna(subset=["Crash Date", "Crash Year", "City"]).copy()
    dated["Crash Date"] = pd.to_datetime(dated["Crash Date"]).dt.normalize()
    dated["Crash Year"] = dated["Crash Year"].astype(int)
    first, last, complete = {}, {}, []
    all_cities = set(data["City"].dropna().astype(str).unique())

    for year, year_rows in dated.groupby("Crash Year"):
        city_groups = list(year_rows.groupby("City", observed=True))
        first[year] = max(group["Crash Date"].min() for _, group in city_groups)
        last[year] = min(group["Crash Date"].max() for _, group in city_groups)
        if year >= today.year or set(str(city) for city, _ in city_groups) != all_cities:
            continue

        jan_1 = pd.Timestamp(year=year, month=1, day=1)
        dec_31 = pd.Timestamp(year=year, month=12, day=31)
        full_coverage = True
        for _, group in city_groups:
            days = pd.DatetimeIndex(group["Crash Date"].dropna().unique()).sort_values()
            if (days[0] - jan_1).days > YEAR_EDGE_TOLERANCE_DAYS:
                full_coverage = False
                break
            if (dec_31 - days[-1]).days > YEAR_EDGE_TOLERANCE_DAYS:
                full_coverage = False
                break
            if len(set(days.month)) != 12:
                full_coverage = False
                break
            if len(days) > 1 and max((days[1:] - days[:-1]).days) > MAX_UNREPORTED_GAP_DAYS:
                full_coverage = False
                break
        if full_coverage:
            complete.append(year)
    return first, last, sorted(complete)


def sample_crash_points(coords, limit=8000):
    """Retain all fatalities, sample only other crashes, render fatalities last."""
    fatal = coords.loc[coords["Severity"] == "Fatal"]
    nonfatal = coords.loc[coords["Severity"] != "Fatal"]
    budget = max(limit - len(fatal), 0)
    if len(nonfatal) > budget:
        nonfatal = nonfatal.sample(n=budget, random_state=42)
    return pd.concat([nonfatal, fatal], ignore_index=True)
