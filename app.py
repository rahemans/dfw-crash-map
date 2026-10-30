"""
DFW Crash Map | Dallas–Fort Worth traffic crash analytics

Run with:  streamlit run app.py
Needs:     streamlit (recent), pandas, plotly, pydeck, numpy, pyarrow (optional, for fast reloads)
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
import streamlit.components.v1 as components

from crash_logic import compute_year_bounds, sample_crash_points

# --------------------------------------------------
# Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="DFW Crash Map | Dallas–Fort Worth Crash Analytics",
    page_icon="💥",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
CITY_FILES = {  # city label -> TxDOT CRIS export. A missing file is skipped.
    "Dallas": DATA_DIR / "dallas_crashes_2024_2026.csv",
    "Fort Worth": DATA_DIR / "forthworth_crashes_2024_2026.csv",
}
BOTH_LABEL = "Dallas + Fort Worth"
CACHE_VERSION = 2  # bump whenever clean() changes so the cached Parquet file is rebuilt
PARQUET_PATH = DATA_DIR / f"dfw_crashes_clean_v{CACHE_VERSION}.parquet"  # auto-generated cache

MAP_HEIGHT = 480            # height of the embedded map in px
MAP_FIT_SIZE = (660, 480)   # visible map size (px) assumed when choosing the starting zoom
# Bounding box covering both cities, used to drop mis-geocoded points on the map.
MAP_LAT_RANGE = (32.4, 33.2)
MAP_LON_RANGE = (-97.7, -96.3)
FALLBACK_HEADER_ROW = 11  # used only if the header row can't be detected

MAX_SCATTER_POINTS = 8000
HEATMAP_CELL_SIZE = 0.002
MIN_HOUR_SAMPLE = 30  # hide injury-share bars for hours with fewer crashes than this

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
TIME_PERIODS = {
    "Overnight (12 AM–5 AM)": list(range(0, 6)),
    "Morning Rush (6 AM–9 AM)": list(range(6, 10)),
    "Midday (10 AM–3 PM)": list(range(10, 16)),
    "Evening Rush (4 PM–7 PM)": list(range(16, 20)),
    "Late Evening (8 PM–11 PM)": list(range(20, 24)),
}

# One chart palette across the app; warm hues are reserved for severity.
COLORS = {"primary": "#6D9FC4", "secondary": "#94A3B8", "injury": "#D4A15B", "fatal": "#CE7777"}
SEVERITY_RGB = {
    "Fatal": (206, 119, 119),
    "Injury": (212, 161, 91),
    "No injury reported": (109, 159, 196),
}

# Restrained editorial styling. Kept theme-neutral and free of internal data-testid hacks
# (except the metric card, which is stable) so it survives Streamlit upgrades.
st.markdown(
    """
<style>
/* Tighter page: less side padding, no oversized title/section gaps. */
.block-container {
    max-width: 1360px;
    padding: 3rem 2rem 2.5rem 2rem;
}
h1, h2, h3 {
    letter-spacing: -.035em;
    font-weight: 620;
}
h1 {
    font-size: 2.3rem !important;
    padding: 0 0 .1rem 0 !important;
    margin: 0 !important;
}
h3 {
    font-size: 1.25rem !important;
    padding: .2rem 0 .4rem 0 !important;
}
[data-testid="stMetric"] {
    min-height: 128px;
    box-sizing: border-box;
    padding: 1.35rem 1.1rem;
    border: 1px solid rgba(150,160,175,.25);
    border-radius: 7px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
/* Larger KPI typography to use the card area, without increasing card height. */
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] {
    font-size: 1.12rem !important;
    font-weight: 550 !important;
    line-height: 1.4 !important;
}
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] > div {
    font-size: 2.05rem !important;
    font-weight: 700 !important;
    letter-spacing: -.035em;
    line-height: 1.2 !important;
}
.facts {
    margin-top: .5rem;
}
.stat {
    padding: .85rem 0 .8rem;
    border-top: 1px solid rgba(150,160,175,.22);
}
.stat-label {
    font-size: .72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .07em;
    opacity: .6;
    margin-bottom: .3rem;
}
.stat-value {
    font-size: 1.3rem;
    font-weight: 700;
    letter-spacing: -.02em;
    line-height: 1.2;
    margin-bottom: .25rem;
}
.stat-note {
    font-size: .82rem;
    opacity: .65;
    line-height: 1.35;
}
.facts-note {
    font-size: .8rem;
    opacity: .6;
    line-height: 1.45;
    padding-top: .8rem;
    border-top: 1px solid rgba(150,160,175,.22);
}
hr {
    margin: .9rem 0 !important;
}
/* Compact sidebar: shrink the empty header strip and the gaps between groups. */
/* Sidebar arrows live in the bottom corners instead of a top strip.
   Collapse arrow: bottom-right of the open sidebar. Expand arrow: bottom-left of the page. */
[data-testid="stSidebar"] {
    position: relative;
}
[data-testid="stSidebarHeader"] {
    position: absolute !important; top: auto !important; left: auto !important;
    bottom: .5rem !important; right: .5rem !important;
    width: 2.6rem !important; height: 2.6rem !important; min-height: 0 !important;
    padding: 0 !important; margin: 0 !important; z-index: 100 !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    border-radius: .5rem; background: rgba(150,160,175,.18); backdrop-filter: blur(4px);
}
[data-testid="stSidebarCollapseButton"] {
    margin: 0 !important; width: 100%; height: 100%;
    display: flex !important; align-items: center; justify-content: center;
}
[data-testid="stSidebarCollapseButton"] button, button[data-testid="stExpandSidebarButton"] {
    margin: 0 !important; padding: 0 !important; width: 100%; height: 100%;
    display: flex !important; align-items: center; justify-content: center;
    opacity: 1 !important; visibility: visible !important;
}
button[data-testid="stExpandSidebarButton"] {
    position: fixed !important; top: auto !important; left: .5rem !important; bottom: .5rem !important;
    width: 2.6rem !important; height: 2.6rem !important; z-index: 1000 !important;
    border-radius: .5rem; background: rgba(150,160,175,.18); backdrop-filter: blur(4px);
}
[data-testid="stSidebarUserContent"] {
    padding-top: .9rem;
    padding-bottom: 3.4rem;
}
[data-testid="stSidebar"] hr {
    margin: .85rem 0 1rem !important;
}
[data-testid="stSidebar"] h3 {
    font-size: 1.35rem !important;
    padding: 0 0 .2rem 0 !important;
}
/* Slightly larger sidebar filter labels and selected values. */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] label p {
    font-size: 1.25rem !important;
    font-weight: 600 !important;
    line-height: 1.45 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] {
    font-size: 1.17rem !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="select"] div {
    font-size: 1.17rem !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] p {
    font-size: 1.12rem !important;
}
[data-testid="stSidebar"] button[kind="secondary"] p {
    font-size: 1.12rem !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] * {
    font-size: inherit;
}
/* Make tabs obvious: larger, bold, boxed. */
[data-baseweb="tab-list"] {
    gap: .5rem;
}
button[data-baseweb="tab"] {
    height: auto;
    padding: .6rem 1.1rem;
    border-radius: 6px 6px 0 0;
    background: rgba(150,160,175,.12);
}
button[data-baseweb="tab"] p {
    font-size: 1.05rem;
    font-weight: 650;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background: rgba(109,159,196,.22);
}
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Labels and formatting helpers
# --------------------------------------------------
KNOWN_STREET_NAMES = {
    "IH0035E": "I-35E",
    "IH0030": "I-30",
    "IH0635": "I-635 (LBJ Freeway)",
    "SL0012": "Loop 12",
    "IH0020": "I-20",
    "US0075": "US-75 (Central Expressway)",
    "DALLAS NORTH TLWY": "Dallas North Tollway",
    "IH0045": "I-45",
    "US0175": "US-175",
    "SS0366": "Spur 366 (Woodall Rodgers Freeway)",
}
HIGHWAY_PREFIXES = {
    "IH": "I-", "US": "US-", "SH": "SH-",
    "SL": "Loop ", "SS": "Spur ",
    "FM": "FM ", "RM": "RM ", "BU": "Business US-",
}
HIGHWAY_PATTERN = re.compile(r"(IH|US|SH|SL|SS|FM|RM|BU)(0*[0-9]+)([A-Z]?)")
OTHER_CODE_PATTERN = re.compile(r"[A-Z]{2}0*[0-9]{2,}[A-Z]?")


def format_street_name(name):
    """Convert TxDOT roadway identifiers into familiar display names."""
    if pd.isna(name):
        return "Unknown"
    raw = re.sub(r"\s+", " ", str(name).strip())
    code = raw.upper()
    if code in KNOWN_STREET_NAMES:
        return KNOWN_STREET_NAMES[code]
    match = HIGHWAY_PATTERN.fullmatch(code)
    if match:
        prefix, number, suffix = match.groups()
        return f"{HIGHWAY_PREFIXES[prefix]}{int(number)}{suffix}"
    if OTHER_CODE_PATTERN.fullmatch(code):
        return code  # unrecognised TxDOT code: show as-is rather than a mangled title-case
    return raw.title()


def hour_label(hour):
    hour = int(hour)
    return f"{(hour % 12) or 12}:00 {'AM' if hour < 12 else 'PM'}"


def short_hour_label(hour):
    hour = int(hour)
    return f"{(hour % 12) or 12} {'AM' if hour < 12 else 'PM'}"


def fmt_date(d):
    # Avoids the platform-specific "%-d" strftime flag.
    return f"{d:%B} {d.day}, {d.year}"


def weather_label(value):
    return value.split(" - ", 1)[-1].title() if " - " in value else value.title()


def road_label(value):
    return value.replace("_", " ").title()


def metric_color(label):
    low = label.lower()
    if "fatal" in low:
        return COLORS["fatal"]
    if "injur" in low:
        return COLORS["injury"]
    return COLORS["primary"]


# --------------------------------------------------
# Data loading
# --------------------------------------------------
def find_header_row(path, default=FALLBACK_HEADER_ROW):
    """CRIS exports start with a preamble; find the real header line instead of hardcoding it."""
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for index, line in enumerate(handle):
                if index > 80:
                    break
                if "Crash Date" in line and "Latitude" in line and line.count(",") >= 5:
                    return index
    except OSError:
        pass
    return default


def clean(raw):
    data = raw.copy()
    data["Crash Date"] = pd.to_datetime(data["Crash Date"], errors="coerce")
    for column in ["Crash Year", "Crash Month", "Crash Total Injury Count", "Crash Death Count", "Latitude", "Longitude"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    raw_hours = data["Hour of Day"].astype("string").str.strip()
    numeric_hours = pd.to_numeric(raw_hours, errors="coerce")
    range_hours = pd.to_numeric(
        raw_hours.str.extract(r"^(\d{1,2}):\d{2}", expand=False),
        errors="coerce",
    )
    data["Hour of Day"] = numeric_hours.fillna(range_hours)
    data.loc[~data["Hour of Day"].between(0, 23), "Hour of Day"] = float("nan")

    for column in ["Weather Condition", "Road Class"]:
        data[column] = data[column].fillna("Unknown").astype(str).astype("category")

    # Derive weekdays from the date, avoiding inconsistently formatted day labels.
    data["Weekday"] = pd.Categorical(
        data["Crash Date"].dt.day_name(),
        categories=DAY_ORDER,
        ordered=True,
    )
    data["Injury Crash"] = data["Crash Total Injury Count"].gt(0)
    data["Fatal Crash"] = data["Crash Death Count"].gt(0)

    # Normalize street identifiers, then derive display names. Grouping on the *display*
    # name merges aliases such as "SH0183" / "SH183" that map to the same label.
    street = data["Street Name"].astype("string").str.strip().str.upper()
    street = street.str.replace(r"\s+", " ", regex=True)
    street = street.mask(street.isin(["", "UNKNOWN", "UNNAMED", "N/A", "NONE", "NAN"]))
    street = street.astype(object).where(street.notna(), np.nan)
    data["Street Clean"] = street
    mapping = {value: format_street_name(value) for value in street.dropna().unique()}
    data["Street Display"] = street.map(mapping)
    return data


def data_signature():
    """Changes whenever a source CSV is added, removed or modified, so the cache reloads."""
    return tuple(
        (city, path.stat().st_mtime, path.stat().st_size)
        for city, path in CITY_FILES.items() if path.exists()
    )


@st.cache_resource(show_spinner="Loading crash records...", max_entries=2)
def load_data(signature):
    """Returns the shared base dataframe for every city. Treat it as read-only (never mutate it).
    `signature` is only a cache key (see data_signature)."""
    available = {city: path for city, path in CITY_FILES.items() if path.exists()}
    newest_csv = max((path.stat().st_mtime for path in available.values()), default=0)

    if PARQUET_PATH.exists() and PARQUET_PATH.stat().st_mtime >= newest_csv:
        try:
            cached = pd.read_parquet(PARQUET_PATH)
            if not available or set(cached["City"].astype(str).unique()) == set(available):
                return cached
        except Exception:
            pass  # fall through and rebuild from the CSVs

    if not available:
        return None

    frames = []
    for city, path in available.items():
        raw = pd.read_csv(path, skiprows=find_header_row(path), low_memory=False)
        raw["City"] = city
        frames.append(raw)
    data = clean(pd.concat(frames, ignore_index=True))
    data["City"] = data["City"].astype("category")
    try:
        data.to_parquet(PARQUET_PATH, index=False)
    except Exception:
        pass  # pyarrow missing or folder read-only: the app still works, just reloads slower
    return data




# --------------------------------------------------
# Filtering and cached aggregates
# --------------------------------------------------
def apply_filters(data, year, weather, roads, periods, include_year=True):
    filtered = data
    if include_year:
        filtered = filtered.loc[filtered["Crash Year"] == year]
    if weather:
        filtered = filtered.loc[filtered["Weather Condition"].isin(weather)]
    if roads:
        filtered = filtered.loc[filtered["Road Class"].isin(roads)]
    if periods:
        hours = set()
        for period in periods:
            hours.update(TIME_PERIODS[period])
        filtered = filtered.loc[filtered["Hour of Day"].isin(hours)]
    return filtered


@st.cache_data(show_spinner=False, max_entries=24)
def map_data(_data, city, year, weather, roads, periods, severity):
    """Geocoded crashes inside the DFW bounding box. `_data` and `city` identify the cached result."""
    subset = apply_filters(_data, year, weather, roads, periods)
    if severity == "Injury Crashes":
        subset = subset.loc[subset["Injury Crash"]]
    elif severity == "Fatal Crashes":
        subset = subset.loc[subset["Fatal Crash"]]
    subset = subset.loc[
        subset["Latitude"].between(*MAP_LAT_RANGE) & subset["Longitude"].between(*MAP_LON_RANGE)
    ]
    if subset.empty:
        return pd.DataFrame(columns=["lat", "lon", "Severity", "Street", "Date", "Injuries", "Deaths", "Injury Detail", "Fatality Detail", "r", "g", "b"])

    level = np.select(
        [subset["Fatal Crash"].to_numpy(), subset["Injury Crash"].to_numpy()],
        ["Fatal", "Injury"],
        default="No injury reported",
    )
    out = pd.DataFrame({
        "lat": subset["Latitude"].to_numpy(),
        "lon": subset["Longitude"].to_numpy(),
        "Severity": level,
        "Street": subset["Street Display"].fillna("Unknown").to_numpy(),
        "Date": subset["Crash Date"].dt.strftime("%b %d, %Y").fillna("Unknown date").to_numpy(),
        "Injuries": subset["Crash Total Injury Count"].to_numpy(),
        "Deaths": subset["Crash Death Count"].to_numpy(),
    })
    # Format missing counts explicitly rather than treating missing data as zero.
    def count_detail(value, singular, plural):
        if pd.isna(value):
            return f"{plural.capitalize()}: unknown"
        count = int(value)
        return f"{count:,} {singular if count == 1 else plural}"

    out["Injury Detail"] = out["Injuries"].map(lambda n: count_detail(n, "person injured", "people injured"))
    out["Fatality Detail"] = out["Deaths"].map(lambda n: count_detail(n, "fatality", "fatalities"))
    out["r"] = out["Severity"].map(lambda s: SEVERITY_RGB[s][0])
    out["g"] = out["Severity"].map(lambda s: SEVERITY_RGB[s][1])
    out["b"] = out["Severity"].map(lambda s: SEVERITY_RGB[s][2])
    return out


@st.cache_data(show_spinner=False, max_entries=24)
def heatmap_cells(coords):
    if coords.empty:
        return pd.DataFrame(columns=["lat", "lon", "weight"])
    cells = pd.DataFrame({
        "lat_cell": (coords["lat"] / HEATMAP_CELL_SIZE).round().astype(int),
        "lon_cell": (coords["lon"] / HEATMAP_CELL_SIZE).round().astype(int),
    })
    counts = cells.groupby(["lat_cell", "lon_cell"], sort=False).size().reset_index(name="weight")
    counts["lat"] = counts["lat_cell"] * HEATMAP_CELL_SIZE
    counts["lon"] = counts["lon_cell"] * HEATMAP_CELL_SIZE
    return counts[["lat", "lon", "weight"]]


def weekday_table(data, year, first_dates, last_dates):
    """Crashes per weekday, normalised by how many of each weekday fall in the year's date range."""
    start = pd.Timestamp(year, 1, 1)  # count from Jan 1 so partial-year ranges stay consistent
    end = min(pd.Timestamp(year, 12, 31), last_dates.get(year, pd.Timestamp(year, 12, 31)))
    days = pd.Series(pd.date_range(start, end).day_name()).value_counts()
    counts = data["Weekday"].astype(str).value_counts().reindex(DAY_ORDER, fill_value=0)
    table = pd.DataFrame({"Weekday": DAY_ORDER, "Crashes": counts.to_numpy()})
    table["Days in period"] = table["Weekday"].map(days).fillna(0).astype(int)
    table["Per day"] = np.where(
        table["Days in period"] > 0, table["Crashes"] / table["Days in period"].replace(0, np.nan), np.nan
    )
    return table


# --------------------------------------------------
# Chart helpers
# --------------------------------------------------
def plot_style(fig, height=330, horizontal=False, color=COLORS["primary"]):
    """Shared layout. Fonts/colors inherit from Streamlit's theme so light and dark both work."""
    fig.update_layout(
        height=height,
        margin=dict(l=4, r=18, t=12, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
        showlegend=False,
        bargap=0.27,
    )
    fig.update_xaxes(showgrid=horizontal, gridcolor="rgba(140,150,165,.18)", zeroline=False,
                     linecolor="rgba(140,150,165,.35)")
    fig.update_yaxes(showgrid=not horizontal, gridcolor="rgba(140,150,165,.18)", zeroline=False)
    if fig.data and fig.data[0].type == "bar":
        fig.update_traces(marker_color=color)
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# --------------------------------------------------
# Load data + shared state
# --------------------------------------------------
df_all = load_data(data_signature())
if df_all is None:
    expected = ", ".join(f"`{path}`" for path in CITY_FILES.values())
    st.error(f"No data files found. Place the TxDOT CRIS exports at: {expected}")
    st.stop()

loaded_cities = set(df_all["City"].astype(str))
CITIES = [city for city in CITY_FILES if city in loaded_cities]
CITY_CHOICES = ([BOTH_LABEL] if len(CITIES) > 1 else []) + CITIES


def city_frame(choice):
    return df_all if choice == BOTH_LABEL else df_all.loc[df_all["City"] == choice]


def years_of(frame):
    return sorted(frame["Crash Year"].dropna().astype(int).unique().tolist())


FILTER_KEYS = {"weather": "f_weather", "roads": "f_roads", "periods": "f_periods"}


def init_state_from_query_params():
    """Seed widget state from the URL once per session so views can be shared as links."""
    if st.session_state.get("_initialized"):
        return
    params = st.query_params
    requested_city = params.get("city", CITY_CHOICES[0])
    city = requested_city if requested_city in CITY_CHOICES else CITY_CHOICES[0]
    st.session_state["city"] = city
    city_years = years_of(city_frame(city))
    default_year = 2025 if 2025 in city_years else city_years[-1]
    try:
        requested = int(params.get("year", default_year))
    except (TypeError, ValueError):
        requested = default_year
    st.session_state["year"] = requested if requested in city_years else default_year
    for name, key in FILTER_KEYS.items():
        st.session_state[key] = params.get_all(name)
    st.session_state["_initialized"] = True


def reset_filters():
    for key in FILTER_KEYS.values():
        st.session_state[key] = []


def sync_query_params(city, year, weather, roads, periods):
    wanted = {"city": city, "year": str(year), "weather": list(weather), "roads": list(roads), "periods": list(periods)}
    for key, value in wanted.items():
        if value:
            if st.query_params.get_all(key) != (value if isinstance(value, list) else [value]):
                st.query_params[key] = value
        elif key in st.query_params:
            del st.query_params[key]


init_state_from_query_params()

# The city selector lives in the sidebar, but its value is already in session state here.
city_choice = st.session_state.get("city", CITY_CHOICES[0])
if city_choice not in CITY_CHOICES:
    city_choice = st.session_state["city"] = CITY_CHOICES[0]
df = city_frame(city_choice)
region_name = "Dallas–Fort Worth" if city_choice == BOTH_LABEL else city_choice

years = years_of(df)
if st.session_state.get("year") not in years:
    st.session_state["year"] = 2025 if 2025 in years else years[-1]
first_dates, last_dates, complete_years = compute_year_bounds(df)

# --------------------------------------------------
# Sidebar filters
# --------------------------------------------------
with st.sidebar:
    st.subheader("Filters")
    st.divider()
    st.selectbox("City", CITY_CHOICES, key="city")
    selected_year = st.selectbox("Crash year", years, key="year")

    year_data = df.loc[df["Crash Year"] == selected_year]
    weather_options = sorted(year_data["Weather Condition"].astype(str).unique().tolist())
    road_options = sorted(year_data["Road Class"].astype(str).unique().tolist())
    period_options = list(TIME_PERIODS)

    # Drop stale selections (e.g. after switching years) before the widgets are created.
    for key, options in [("f_weather", weather_options), ("f_roads", road_options), ("f_periods", period_options)]:
        st.session_state[key] = [v for v in st.session_state.get(key, []) if v in options]

    selected_weather = st.multiselect("Weather conditions", weather_options, format_func=weather_label,
                                      key="f_weather", placeholder="All conditions")
    selected_roads = st.multiselect("Road types", road_options, format_func=road_label,
                                    key="f_roads", placeholder="All road types")
    selected_periods = st.multiselect("Time of day", period_options,
                                      key="f_periods", placeholder="All times")
    st.button("Reset filters", on_click=reset_filters, width="stretch")

    st.divider()
    layout_mode = st.radio("Page layout", ["Single page", "Tabs"], horizontal=True, key="layout",
                           help="Single page shows every section in one scroll. Tabs show one section at a time.")

    st.caption(f"Source: TxDOT CRIS · {region_name}")
    partial_years = [y for y in years if y not in complete_years]
    for y in partial_years:
        if city_choice == BOTH_LABEL:
            cutoffs = df.loc[df["Crash Year"] == y].groupby("City", observed=True)["Crash Date"].max()
            parts = "; ".join(f"{city} through {fmt_date(day)}" for city, day in cutoffs.items() if pd.notna(day))
            st.caption(f"{y} has partial or unverified calendar coverage: {parts}.")
        elif y in last_dates:
            st.caption(f"{y} crash records through {fmt_date(last_dates[y])} (partial or unverified calendar coverage).")
    st.caption("Map: geocoded crashes only.")

weather = tuple(sorted(selected_weather))
roads = tuple(sorted(selected_roads))
periods = tuple(sorted(selected_periods))
sync_query_params(city_choice, selected_year, weather, roads, periods)

filtered = apply_filters(df, selected_year, weather, roads, periods)
if city_choice == BOTH_LABEL:
    # The same street name can exist in both cities, so qualify labels with the city.
    filtered = filtered.assign(**{"Street Label": filtered["Street Display"] + " (" + filtered["City"].astype(str) + ")"})
    street_col = "Street Label"
else:
    street_col = "Street Display"
all_years_filtered = apply_filters(df, None, weather, roads, periods, include_year=False)

# --------------------------------------------------
# Map
# --------------------------------------------------
def fit_view(lat, lon, margin=1.12, size=MAP_FIT_SIZE):
    """Pick a camera that frames the crashes being shown.

    Uses the 1st-99th percentile so a few stray points don't zoom the map out, and deck.gl's
    512-px world size. `size` is the map size (px) the framing is guaranteed to fit.
    """
    lo_lat, hi_lat = np.percentile(lat, [1, 99])
    lo_lon, hi_lon = np.percentile(lon, [1, 99])

    def mercator(deg):
        return np.log(np.tan(np.pi / 4 + np.radians(deg) / 2))

    x_fraction = max(hi_lon - lo_lon, 0.02) * margin / 360
    y_fraction = max(abs(mercator(hi_lat) - mercator(lo_lat)), 0.0004) * margin / (2 * np.pi)
    zoom = min(np.log2(size[0] / (512 * x_fraction)), np.log2(size[1] / (512 * y_fraction)))
    return float((lo_lat + hi_lat) / 2), float((lo_lon + hi_lon) / 2), float(np.clip(zoom, 8, 14))


# Camera memory. Streamlit replaces the whole map document whenever a map option changes
# (All/Injury/Fatal, Heatmap/Points, a sidebar filter), which would normally snap the map back
# to its starting view. The embedding page's `window.name` survives that swap (a full page
# reload clears it), so the map saves its camera there whenever the user moves it and restores
# it on load. "Reset map view" clears it. A saved camera is only reused for the same
# `context` (the selected city), so switching city re-frames the map.
CAMERA_RESTORE_JS = """
    const MAP_CONTEXT = __CONTEXT__;
    const MAP_STORE = 'dfwmap:';
    try {
      const raw = window.parent.name || '';
      if (raw.indexOf(MAP_STORE) === 0) {
        const saved = JSON.parse(raw.slice(MAP_STORE.length));
        if (saved && saved.ctx === MAP_CONTEXT && saved.view) {
          Object.assign(jsonInput.initialViewState, saved.view);
        }
      }
    } catch (e) {}
    const mapStart = Object.assign({}, jsonInput.initialViewState);

"""
CAMERA_CAPTURE_JS = """
    (function () {
      let lastSaved = '';
      function remember() {
        try {
          const vp = deckInstance.getViewports()[0];
          if (!vp) return;
          const view = {latitude: vp.latitude, longitude: vp.longitude, zoom: vp.zoom,
                        pitch: vp.pitch || 0, bearing: vp.bearing || 0};
          const moved = Math.abs(view.latitude - mapStart.latitude) > 1e-7 ||
                        Math.abs(view.longitude - mapStart.longitude) > 1e-7 ||
                        Math.abs(view.zoom - mapStart.zoom) > 1e-7;
          if (!moved) return;  // only remember a camera the user actually moved
          const text = JSON.stringify(view);
          if (text === lastSaved) return;
          lastSaved = text;
          window.parent.name = MAP_STORE + JSON.stringify({ctx: MAP_CONTEXT, view: view});
        } catch (e) {}
      }
      setInterval(remember, 250);
      window.addEventListener('pagehide', remember);
    })();
"""


def remember_camera(deck_html, context):
    """Add the camera-memory scripts to pydeck's generated page (see note above)."""
    marker = "const deckInstance = createDeck("
    if marker not in deck_html or "</script>" not in deck_html:
        return deck_html  # unfamiliar pydeck template: leave the map exactly as pydeck made it
    restore = CAMERA_RESTORE_JS.replace("__CONTEXT__", json.dumps(context))
    deck_html = deck_html.replace(marker, restore + "    " + marker, 1)
    end = deck_html.rindex("</script>")
    return deck_html[:end] + CAMERA_CAPTURE_JS + "  " + deck_html[end:]


# The map lives in a nested iframe so the Reset button can reload it (which restores the camera).
MAP_EMBED_TEMPLATE = """
<!doctype html>
<html>
<head>
  <style>
    html, body { margin: 0; padding: 0; height: 100%; overflow: hidden; background: #0e1117;
                 font-family: Arial, sans-serif; }
    body { position: relative; }
    iframe { display: block; width: 100%; height: 100%; border: 0; }
    .reset { position: absolute; top: 10px; right: 10px; z-index: 5; background: rgba(32,41,56,.92);
             color: #e7edf6; border: 1px solid #526073; border-radius: 5px; padding: 6px 12px;
             cursor: pointer; font-size: 13px; }
    .reset:hover { background: #304057; }
    .legend { position: absolute; left: 10px; bottom: 10px; z-index: 5; display: flex; gap: 14px;
              background: rgba(14,17,23,.85); color: #d5dce6; border-radius: 5px; padding: 6px 11px;
              font-size: 12px; pointer-events: none; }
    .legend i { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 5px; }
  </style>
</head>
<body>
  <iframe id="crash-map" title="Crash map" srcdoc="__DECK__"></iframe>
  <button class="reset" type="button" onclick="resetView()">Reset map view</button>
  __LEGEND__
  <script>
    // The only thing that returns the map to its starting view: forget the saved camera, reload.
    function resetView() {
      try { window.name = ''; } catch (e) {}
      const frame = document.getElementById('crash-map');
      frame.srcdoc = frame.srcdoc;
    }
  </script>
</body>
</html>
"""


@st.cache_data(show_spinner=False, max_entries=16)
def build_map(_data, city, year, weather, roads, periods, severity, map_style):
    """Builds the full map HTML once per unique selection, so unrelated reruns reuse it."""
    coords = map_data(_data, city, year, weather, roads, periods, severity)
    if coords.empty:
        return None

    legend = ""
    if map_style == "Hotspot Heatmap":
        points = heatmap_cells(coords)
        layer = pdk.Layer("HeatmapLayer", data=points, get_position="[lon, lat]",
                          get_weight="weight", radius_pixels=35, opacity=0.7)
        caption = (f"{len(coords):,} geocoded crashes shown as {len(points):,} weighted grid cells. "
                   "Hotspots show where reported crashes concentrate, not risk per driver or mile.")
        tooltip = None
    else:
        # Always retain every fatal crash. Sample nonfatal crashes only when necessary.
        # Draw fatal markers last so nearby nonfatal markers cannot cover them.
        points = sample_crash_points(coords, limit=MAX_SCATTER_POINTS)
        # A slightly larger marker makes fatal crashes easier to distinguish.
        points["marker_radius"] = np.where(points["Severity"] == "Fatal", 42, 25)
        layer = pdk.Layer(
            "ScatterplotLayer", data=points, get_position="[lon, lat]",
            get_radius="marker_radius", radius_min_pixels=2, get_fill_color="[r, g, b, 170]",
            pickable=True, auto_highlight=True,
        )
        caption = (f"Showing {len(points):,} of {len(coords):,} geocoded crashes "
                   "(all fatal crashes included; nonfatal crashes sampled when necessary). "
                   "Hover a point for injury and fatality counts. Concentration is not risk per driver or mile.")
        tooltip = {"html": "<b>{Street}</b><br/>{Date}<br/>{Severity}<br/>{Injury Detail}<br/>{Fatality Detail}"}
        legend = '<div class="legend">' + "".join(
            f'<span><i style="background: rgb({r},{g},{b})"></i>{label}</span>'
            for label, (r, g, b) in SEVERITY_RGB.items()
        ) + "</div>"

    view_lat, view_lon, view_zoom = fit_view(coords["lat"].to_numpy(), coords["lon"].to_numpy())
    deck_kwargs = dict(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=view_lat, longitude=view_lon, zoom=view_zoom, pitch=0),
        map_style="dark",
    )
    if tooltip:  # passing tooltip=None can break the embedded map, so only set it when needed
        deck_kwargs["tooltip"] = tooltip
    deck = pdk.Deck(**deck_kwargs)
    deck_html = remember_camera(deck.to_html(as_string=True, notebook_display=False), city)
    escaped_deck_html = html.escape(deck_html, quote=True)
    embed = MAP_EMBED_TEMPLATE.replace("__DECK__", escaped_deck_html).replace("__LEGEND__", legend)
    return embed.strip(), caption


@st.fragment
def render_map(data, city, year, weather, roads, periods):
    """Runs as a fragment: changing the map options reruns only this function, not the whole page."""
    left, right = st.columns([3, 2])
    severity = left.segmented_control(
        "Show crashes", ["All Crashes", "Injury Crashes", "Fatal Crashes"],
        format_func=lambda v: v.replace(" Crashes", ""),
        default="All Crashes", key="map_severity",
        help="Injury means at least one reported injury; Fatal means at least one reported death. A crash may appear in both categories.",
    )
    map_style = right.segmented_control(
        "Map visualization", ["Hotspot Heatmap", "Individual Crashes"],
        format_func=lambda v: {"Hotspot Heatmap": "Heatmap", "Individual Crashes": "Points"}[v],
        default="Individual Crashes", key="map_style",
        help="Heatmap shows concentration; Points shows individual crashes colored by severity.",
    )
    # A segmented control can be switched off by clicking the active option; fall back to the defaults.
    severity = severity or "All Crashes"
    map_style = map_style or "Individual Crashes"

    result = build_map(data, city, year, weather, roads, periods, severity, map_style)
    if result is None:
        st.info("No valid crash coordinates match these filters.")
        return
    map_embed, caption = result
    if hasattr(st, "iframe"):  # st.components.v1.html is deprecated in recent Streamlit releases
        st.iframe(map_embed, height=MAP_HEIGHT)
    else:
        components.html(map_embed, height=MAP_HEIGHT, scrolling=False)
    st.caption(caption)


# --------------------------------------------------
# Header and overview metrics
# --------------------------------------------------
st.title("DFW Crash Map")
st.caption(f"{region_name} · {selected_year} crash records · every chart follows the sidebar filters")
if filtered.empty:
    st.warning("No crashes match the current filters. Try removing some filters or choosing another year.")
    st.stop()

# Quick facts for the selected year and filters.
valid_hours = filtered["Hour of Day"].dropna()
peak_hour = int(valid_hours.value_counts().idxmax()) if not valid_hours.empty else None
peak_count = int((valid_hours == peak_hour).sum()) if peak_hour is not None else 0
peak_text = hour_label(peak_hour) if peak_hour is not None else "No data"

wk = weekday_table(filtered, selected_year, first_dates, last_dates)
wk_valid = wk.dropna(subset=["Per day"])
if not wk_valid.empty and wk_valid["Per day"].max() > 0:
    top_wk = wk_valid.loc[wk_valid["Per day"].idxmax()]
    weekday_text, weekday_note = top_wk["Weekday"], f"{top_wk['Per day']:.1f} crashes per {top_wk['Weekday']} on average"
else:
    weekday_text, weekday_note = "No data", ""

street_counts = filtered[street_col].dropna().value_counts()
top_road = street_counts.index[0] if not street_counts.empty else "No data"
top_road_count = int(street_counts.iloc[0]) if not street_counts.empty else 0

map_col, stats_col = st.columns([3, 1], gap="medium")
with map_col:
    render_map(df, city_choice, selected_year, weather, roads, periods)
with stats_col:
    # Space the four KPI cards vertically alongside the map.
    # Streamlit adds its own element gaps; these smaller spacers work with
    # the taller KPI cards to fill the map height without overshooting.
    st.metric("Reported crashes", f"{len(filtered):,}")
    st.markdown('<div style="height: 9px"></div>', unsafe_allow_html=True)
    st.metric("Crashes with injuries", f"{int(filtered['Injury Crash'].sum()):,}")
    st.markdown('<div style="height: 9px"></div>', unsafe_allow_html=True)
    st.metric("Fatal crashes", f"{int(filtered['Fatal Crash'].sum()):,}")
    st.markdown('<div style="height: 9px"></div>', unsafe_allow_html=True)
    st.metric("Injury crash share", f"{filtered['Injury Crash'].mean() * 100:.1f}%")
st.caption("Injury and fatal crash counts can overlap when a crash involved both injuries and deaths.")
# Place quick insights in a compact row beneath the map and KPI cards.
# This fills the previously empty area under the map without stretching it.
facts = [
    ("Peak crash hour", peak_text, f"{peak_count:,} crashes at this hour"),
    ("Busiest weekday", weekday_text, weekday_note),
    ("Highest-count roadway", top_road, f"{top_road_count:,} crashes"),
]
insight_cols = st.columns(3, gap="medium")
for column, (label, value, note) in zip(insight_cols, facts):
    with column:
        st.markdown(
            '<div class="stat"><div class="stat-label">' + html.escape(label) + '</div>'
            '<div class="stat-value">' + html.escape(str(value)) + '</div>'
            '<div class="stat-note">' + html.escape(note) + '</div></div>',
            unsafe_allow_html=True,
        )
st.caption("Counts reflect reported crashes, not traffic-adjusted risk or road length.")

SECTION_NAMES = (["Dallas vs. Fort Worth"] if city_choice == BOTH_LABEL else []) + [
    "When crashes happen", "Streets & road types", "Year comparison",
]
if layout_mode == "Tabs":
    containers = st.tabs(SECTION_NAMES)
else:
    containers = []
    for _ in SECTION_NAMES:
        st.divider()
        containers.append(st.container())
slots = dict(zip(SECTION_NAMES, containers))
tab_cities = slots.get("Dallas vs. Fort Worth")
tab_time, tab_roads, tab_compare = slots["When crashes happen"], slots["Streets & road types"], slots["Year comparison"]


# --------------------------------------------------
# Dallas vs. Fort Worth (only when both cities are shown)
# --------------------------------------------------
if tab_cities is not None:
    with tab_cities:
        st.subheader("Dallas vs. Fort Worth")
        by_city = (
            filtered.groupby("City", observed=True)
            .agg(Crashes=("Injury Crash", "size"), Injury_Crashes=("Injury Crash", "sum"),
                 Fatal_Crashes=("Fatal Crash", "sum"))
            .reset_index()
        )
        by_city["City"] = by_city["City"].astype(str)
        by_city["Injury share (%)"] = by_city["Injury_Crashes"] / by_city["Crashes"] * 100
        city_left, city_right = st.columns(2, gap="large")
        with city_left:
            st.markdown("**Reported crashes by city**")
            fig = px.bar(by_city, x="City", y="Crashes")
            fig.update_traces(hovertemplate="<b>%{x}</b><br>Crashes: %{y:,}<extra></extra>")
            fig.update_xaxes(title=None)
            show(plot_style(fig, 300))
        with city_right:
            st.markdown("**Injury crash share by city**")
            fig = px.bar(by_city, x="City", y="Injury share (%)")
            fig.update_traces(hovertemplate="<b>%{x}</b><br>Injury crash share: %{y:.1f}%<extra></extra>")
            fig.update_xaxes(title=None)
            fig.update_yaxes(title="Share of reported crashes (%)")
            show(plot_style(fig, 300, color=COLORS["injury"]))
        st.dataframe(
            by_city.rename(columns={"Injury_Crashes": "Injury crashes", "Fatal_Crashes": "Fatal crashes"})
            [["City", "Crashes", "Injury crashes", "Fatal crashes", "Injury share (%)"]]
            .style.format({"Injury share (%)": "{:.1f}"}),
            hide_index=True, width="stretch",
        )
        st.caption("Raw counts for the selected year and filters. The cities differ in population, road mileage and "
                   "reporting practices, so these totals are not a measure of which city is safer.")

# --------------------------------------------------
# Time patterns
# --------------------------------------------------
with tab_time:
    st.subheader("When crashes happen")
    st.caption("Reported crashes across months, hours of the day, and days of the week")

    monthly_counts = (
        all_years_filtered.dropna(subset=["Crash Year", "Crash Month"])
        .groupby(["Crash Year", "Crash Month"]).size().rename("Crashes").reset_index()
    )
    monthly_counts["Crash Year"] = monthly_counts["Crash Year"].astype(int)
    monthly_counts["Crash Month"] = monthly_counts["Crash Month"].astype(int)

    def monthly_series(year):
        """Counts per month; months beyond the data cutoff become NaN (not zero), and a
        partially covered final month is returned separately so it isn't read as a drop."""
        rows = monthly_counts.loc[monthly_counts["Crash Year"] == year].set_index("Crash Month")["Crashes"]
        values = rows.reindex(range(1, 13), fill_value=0).astype(float)
        partial_month = None
        if year not in complete_years and year in last_dates:
            last = last_dates[year]
            values[values.index > last.month] = np.nan
            if last.day < last.days_in_month:
                partial_month = last.month
        line = values.copy()
        if partial_month:
            line[partial_month] = np.nan
        return line, partial_month, (values[partial_month] if partial_month else None)

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Monthly trend**")
        fig = go.Figure()
        prior_year = selected_year - 1
        if prior_year in years:
            prior_line, _, _ = monthly_series(prior_year)
            fig.add_trace(go.Scatter(
                x=MONTH_LABELS, y=prior_line, name=str(prior_year), mode="lines",
                line=dict(color=COLORS["secondary"], dash="dot", width=2), opacity=0.8,
                hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y:,.0f}<extra></extra>",
            ))
        line, partial_month, partial_value = monthly_series(selected_year)
        fig.add_trace(go.Scatter(
            x=MONTH_LABELS, y=line, name=str(selected_year), mode="lines+markers",
            line=dict(color=COLORS["primary"], width=3), marker=dict(size=6),
            hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y:,.0f}<extra></extra>",
        ))
        if partial_month:
            fig.add_trace(go.Scatter(
                x=[MONTH_LABELS[partial_month - 1]], y=[partial_value], name="Partial month", mode="markers",
                marker=dict(size=10, symbol="circle-open", line=dict(width=2, color=COLORS["primary"])),
                hovertemplate="<b>%{x}</b> (partial month)<br>Crashes so far: %{y:,.0f}<extra></extra>",
            ))
        fig = plot_style(fig)
        fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.14, x=0))
        fig.update_xaxes(title=None)
        show(fig)
        if partial_month:
            st.caption(f"{MONTH_LABELS[partial_month - 1]} is only partly covered (data through "
                       f"{fmt_date(last_dates[selected_year])}); the hollow marker shows crashes so far.")
        if prior_year in years:
            st.caption(f"Dotted line: {prior_year}, same filters.")

    with right:
        hourly = filtered.groupby("Hour of Day").size()
        hourly.index = hourly.index.astype(int)
        hourly = hourly.reindex(range(24), fill_value=0).rename("Crashes").reset_index()
        hourly["Label"] = hourly["Hour of Day"].map(hour_label)
        st.markdown("**Crashes by hour**")
        fig = px.bar(hourly, x="Hour of Day", y="Crashes", custom_data=["Label"])
        fig.update_traces(hovertemplate="<b>%{customdata[0]}</b><br>Crashes: %{y:,}<extra></extra>")
        fig.update_xaxes(tickmode="array", tickvals=list(range(0, 24, 3)),
                         ticktext=[short_hour_label(h) for h in range(0, 24, 3)], title=None)
        show(plot_style(fig))
        missing_hours = int(filtered["Hour of Day"].isna().sum())
        if missing_hours:
            st.caption(f"{missing_hours:,} records have unknown or unrecognized hours and are excluded here.")

    st.divider()
    st.subheader("Day of week")
    mode = st.radio(
        "Show",
        ["Average crashes per day", "Total crashes"],
        horizontal=True,
        key="weekday_mode",
    )
    value_col = "Per day" if mode.startswith("Average") else "Crashes"
    wk_chart = weekday_table(filtered, selected_year, first_dates, last_dates)
    fig = px.bar(wk_chart, x="Weekday", y=value_col, custom_data=["Crashes", "Days in period"])
    fig.update_traces(hovertemplate=(
        "<b>%{x}</b><br>" + mode + ": %{y:,.1f}<br>Total: %{customdata[0]:,} over %{customdata[1]} days<extra></extra>"
    ))
    fig.update_xaxes(title=None)
    fig.update_yaxes(title=mode)
    show(plot_style(fig, 340))
    unknown_weekdays = int(filtered["Weekday"].isna().sum())
    if unknown_weekdays:
        st.caption(f"{unknown_weekdays:,} records have no valid crash date and are excluded from the weekday chart.")
    st.caption("Averages divide each weekday's total by how many of that weekday occur in the year's date range. "
               "Counts reflect reported crashes, not risk per trip.")

    st.divider()
    st.subheader("Hour × weekday")
    st.caption("Total reported crashes for each weekday and hour combination")
    timed = filtered.dropna(subset=["Hour of Day", "Weekday"])
    grid = pd.crosstab(timed["Weekday"].astype(str), timed["Hour of Day"].astype(int)).reindex(
        index=DAY_ORDER, columns=range(24), fill_value=0
    )
    hour_names = [short_hour_label(h) for h in range(24)]
    fig = px.imshow(
        grid.to_numpy(), x=hour_names, y=DAY_ORDER, aspect="auto",
        color_continuous_scale=[[0, "rgba(109,159,196,0.07)"], [1, COLORS["primary"]]],
        labels=dict(color="Crashes"),
    )
    fig.update_traces(hovertemplate="<b>%{y}, %{x}</b><br>Crashes: %{z:,}<extra></extra>")
    fig = plot_style(fig, 330)
    fig.update_xaxes(
        tickmode="array",
        tickvals=hour_names[::3],
        ticktext=hour_names[::3],
        title=None,
    )
    fig.update_yaxes(title=None)
    fig.update_layout(coloraxis_colorbar=dict(title="Crashes", thickness=12))
    show(fig)

    st.divider()
    st.subheader("Severity by hour")
    st.caption(f"Share of reported crashes that involved an injury. Hours with fewer than {MIN_HOUR_SAMPLE} crashes are hidden.")
    sev = filtered.dropna(subset=["Hour of Day"]).assign(Hour=lambda d: d["Hour of Day"].astype(int))
    sev = sev.groupby("Hour").agg(Crashes=("Hour", "size"), Injury=("Injury Crash", "sum")).reindex(
        range(24),
        fill_value=0,
    )
    sev["Injury share"] = np.where(
        sev["Crashes"] >= MIN_HOUR_SAMPLE,
        sev["Injury"] / sev["Crashes"].replace(0, np.nan) * 100,
        np.nan,
    )
    sev = sev.reset_index().rename(columns={"index": "Hour"})
    sev["Label"] = sev["Hour"].map(hour_label)
    fig = px.bar(sev, x="Hour", y="Injury share", custom_data=["Label", "Crashes", "Injury"])
    fig.update_traces(hovertemplate=(
        "<b>%{customdata[0]}</b><br>Injury share: %{y:.1f}%<br>"
        "%{customdata[2]:,} of %{customdata[1]:,} crashes<extra></extra>"
    ))
    fig.update_xaxes(tickmode="array", tickvals=list(range(0, 24, 3)),
                     ticktext=[short_hour_label(h) for h in range(0, 24, 3)], title=None)
    fig.update_yaxes(title="Injury crash share (%)")
    show(plot_style(fig, 320, color=COLORS["injury"]))

# --------------------------------------------------
# Streets and road types
# --------------------------------------------------
with tab_roads:
    st.subheader("Crash-heavy streets")
    st.write("Rank streets by reported crash counts, injury crashes, or fatal crashes. "
             "This is a volume ranking, not a risk-adjusted safety ranking.")
    rank_by = st.selectbox(
        "Rank streets by",
        ["Total crashes", "Crashes with injuries", "Fatal crashes"],
        key="street_rank",
    )
    rank_column = {"Total crashes": "Crashes", "Crashes with injuries": "Injury crashes", "Fatal crashes": "Fatal crashes"}[rank_by]

    streets = (
        filtered.dropna(subset=[street_col]).groupby(street_col)
        .agg(Crashes=(street_col, "size"), **{"Injury crashes": ("Injury Crash", "sum"),
                                                    "Fatal crashes": ("Fatal Crash", "sum")})
        .reset_index().rename(columns={street_col: "Street"})
    )
    ranked = streets.loc[streets[rank_column] > 0].sort_values(
        [rank_column, "Crashes", "Street"], ascending=[False, False, True]
    )
    top = ranked.head(10)
    if top.empty:
        st.info("No streets with matching records for this ranking and filter selection.")
    else:
        st.markdown(f"**Top 10 · {rank_by.lower()}**")
        fig = px.bar(
            top.sort_values(rank_column),
            x=rank_column,
            y="Street",
            orientation="h",
            text=rank_column,
        )
        fig.update_traces(hovertemplate="<b>%{y}</b><br>" + rank_by + ": %{x:,}<extra></extra>",
                          textposition="outside", cliponaxis=False)
        fig.update_layout(yaxis_title=None, xaxis_title=rank_by)
        show(plot_style(fig, 410, horizontal=True, color=metric_color(rank_by)))
        st.dataframe(
            top[["Street", "Crashes", "Injury crashes", "Fatal crashes"]],
            hide_index=True,
            width="stretch",
        )
        st.download_button(
            "Download full ranking (CSV)",
            ranked[["Street", "Crashes", "Injury crashes", "Fatal crashes"]].to_csv(index=False).encode("utf-8"),
            file_name=f"{re.sub(r'[^a-z0-9]+', '_', city_choice.lower()).strip('_')}_streets_{selected_year}.csv", mime="text/csv",
        )
    st.caption("TxDOT road codes are translated for readability and aliases that map to the same label are combined. "
               "Different segments or nicknames for one road may still be counted separately. Missing street names are excluded.")

    st.divider()
    st.subheader("Road type analysis")
    road_counts = (
        filtered.groupby("Road Class", observed=True)
        .agg(Crashes=("Road Class", "size"), Injury_Crashes=("Injury Crash", "sum")).reset_index()
    )
    road_counts["Road type"] = road_counts["Road Class"].astype(str).map(road_label)
    road_counts["Injury share"] = road_counts["Injury_Crashes"] / road_counts["Crashes"] * 100
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Crashes by road type**")
        fig = px.bar(
            road_counts.sort_values("Crashes"),
            x="Crashes",
            y="Road type",
            orientation="h",
        )
        fig.update_traces(hovertemplate="<b>%{y}</b><br>Crashes: %{x:,}<extra></extra>")
        fig.update_yaxes(title=None)
        show(plot_style(fig, 390, horizontal=True))
    with right:
        st.markdown("**Injury crash share by road type**")
        fig = px.bar(
            road_counts.sort_values("Injury share"),
            x="Injury share",
            y="Road type",
            orientation="h",
        )
        fig.update_traces(hovertemplate="<b>%{y}</b><br>Injury crash share: %{x:.1f}%<extra></extra>")
        fig.update_xaxes(title="Share of reported crashes (%)")
        fig.update_yaxes(title=None)
        show(plot_style(fig, 390, horizontal=True, color=COLORS["injury"]))

# --------------------------------------------------
# Year-over-year comparison (years derived from the data)
# --------------------------------------------------
with tab_compare:
    st.subheader("Year comparison")
    if len(complete_years) < 2:
        st.info("At least two years passing the data-coverage check are needed for a comparison.")
    else:
        st.write("Compare two years with near-full calendar coverage using the weather, road, and time filters selected in the sidebar.")
        ca, cb = st.columns(2)
        year_a = ca.selectbox(
            "Baseline year",
            complete_years,
            index=len(complete_years) - 2,
            key="cmp_a",
        )
        year_b = cb.selectbox(
            "Comparison year",
            complete_years,
            index=len(complete_years) - 1,
            key="cmp_b",
        )
        if year_a == year_b:
            st.info("Choose two different years.")
        else:
            comp = all_years_filtered.loc[all_years_filtered["Crash Year"].isin([year_a, year_b])]
            summary = comp.groupby("Crash Year").agg(
                Crashes=("Crash Year", "size"),
                Injury_Crashes=("Injury Crash", "sum"),
                Fatal_Crashes=("Fatal Crash", "sum"),
            )
            summary.index = summary.index.astype(int)
            summary = summary.reindex([year_a, year_b], fill_value=0)

            compare_metric = st.radio("Compare", ["Crashes", "Injury crashes", "Fatal crashes"],
                                      horizontal=True, key="cmp_metric")
            metric_column = {"Crashes": "Crashes", "Injury crashes": "Injury_Crashes", "Fatal crashes": "Fatal_Crashes"}[compare_metric]
            chart = summary.reset_index().rename(columns={"Crash Year": "Year"})
            chart["Year"] = chart["Year"].astype(str)

            st.markdown(f"**{compare_metric} · annual comparison**")
            fig = px.bar(chart, x="Year", y=metric_column, text=metric_column)
            fig.update_traces(hovertemplate="<b>%{x}</b><br>" + compare_metric + ": %{y:,}<extra></extra>",
                              textposition="outside")
            fig.update_xaxes(type="category", title=None)
            fig.update_yaxes(title=compare_metric)
            show(plot_style(fig, 330, color=metric_color(compare_metric)))

            old_count = int(summary.loc[year_a, metric_column])
            new_count = int(summary.loc[year_b, metric_column])
            if old_count:
                delta = (new_count - old_count) / old_count * 100
                st.metric(
                    f"{year_b} vs. {year_a} · {compare_metric.lower()}",
                    f"{new_count:,}",
                    f"{delta:+.1f}%",
                    delta_color="off",
                )
                if new_count == old_count:
                    st.caption(f"**Annual insight:** {compare_metric} were unchanged between {year_a} and {year_b} under the selected filters.")
                else:
                    direction = "higher" if new_count > old_count else "lower"
                    st.caption(f"**Annual insight:** {compare_metric} were {abs(delta):.1f}% {direction} in {year_b} than in {year_a} "
                               f"under the selected filters ({old_count:,} → {new_count:,}).")
            else:
                st.caption(f"Percentage change unavailable because the {year_a} baseline is zero.")
            st.caption("Counts reflect reported crashes, not traffic-volume-adjusted risk. Years are screened for coverage near both year boundaries, all 12 months, and long data gaps; this does not independently certify the TxDOT exports as complete.")

st.divider()
st.caption(
    f"Source: Texas Department of Transportation CRIS. Data reflects reported crashes in {region_name}. "
    "Crash concentration is not the same as crash risk per vehicle or mile traveled."
)