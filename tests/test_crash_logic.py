from datetime import date

import pandas as pd

from crash_logic import compute_year_bounds, sample_crash_points


def records_for_year(year, city, days=None):
    days = days if days is not None else pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    return pd.DataFrame({"Crash Date": days, "Crash Year": year, "City": city})


def test_complete_year_for_both_cities():
    df = pd.concat([records_for_year(2024, "Dallas"), records_for_year(2024, "Fort Worth")])
    first, last, complete = compute_year_bounds(df, today=date(2026, 10, 30))
    assert complete == [2024]
    assert first[2024] == pd.Timestamp('2024-01-01')
    assert last[2024] == pd.Timestamp('2024-12-31')


def test_missing_first_ten_days_is_incomplete():
    days = pd.date_range('2024-01-11', '2024-12-31')
    df = records_for_year(2024, 'Dallas', days)
    assert compute_year_bounds(df, today=date(2026, 10, 30))[2] == []


def test_partial_city_excludes_combined_year():
    df = pd.concat([records_for_year(2024, 'Dallas'), records_for_year(2024, 'Fort Worth', pd.date_range('2024-01-01', '2024-09-30'))])
    assert compute_year_bounds(df, today=date(2026, 10, 30))[2] == []


def test_long_gap_is_incomplete():
    days = pd.date_range('2024-01-01', '2024-12-31')
    days = days[(days < '2024-05-01') | (days > '2024-05-24')]
    assert compute_year_bounds(records_for_year(2024, 'Dallas', days), today=date(2026, 10, 30))[2] == []


def test_current_year_never_complete():
    assert compute_year_bounds(records_for_year(2026, 'Dallas'), today=date(2026, 10, 30))[2] == []


def test_fatal_crashes_always_retained_and_last():
    data = pd.DataFrame({'Severity': ['Injury'] * 12000 + ['Fatal'] * 274, 'id': range(12274)})
    sampled = sample_crash_points(data, limit=8000)
    assert len(sampled) == 8000
    assert (sampled['Severity'] == 'Fatal').sum() == 274
    assert sampled['Severity'].iloc[-274:].eq('Fatal').all()
    assert sampled['id'].is_unique


def test_when_fatalities_exceed_limit():
    data = pd.DataFrame({'Severity': ['Fatal'] * 12 + ['Injury'] * 30})
    sampled = sample_crash_points(data, limit=10)
    assert len(sampled) == 12
    assert sampled['Severity'].eq('Fatal').all()


def test_empty_points():
    data = pd.DataFrame({'Severity': pd.Series(dtype='str')})
    assert sample_crash_points(data).empty
