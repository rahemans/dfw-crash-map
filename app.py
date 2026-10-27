"""
DFW Crash Map | Dallas–Fort Worth traffic crash analytics

Run with:  streamlit run app.py
Needs:     streamlit (recent), pandas, plotly, pydeck, numpy, pyarrow (optional, for fast reloads)
"""
from __future__ import annotations

import html
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
import streamlit.components.v1 as components

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

# Starting camera per selection: (latitude, longitude, zoom)
CITY_VIEWS = {
    "Dallas": (32.7767, -96.7970, 11.5),
    "Fort Worth": (32.7555, -97.3308, 11.2),
    BOTH_LABEL: (32.77, -97.06, 9.7),
}
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
.block-container {max-width: 1360px; padding-top: 1.65rem; padding-bottom: 2.75rem;}
h1, h2, h3 {letter-spacing: -.035em; font-weight: 620;}
h1 {font-size: 2.4rem !important; margin-bottom: .15rem !important;}
h3 {font-size: 1.3rem !important;}
[data-testid="stMetric"] {padding: .8rem .9rem; border: 1px solid rgba(150,160,175,.25); border-radius: 5px;}
[data-testid="stMetricValue"] {font-size: 1.7rem; letter-spacing: -.035em;}
hr {margin: 1.4rem 0 !important;}
/* Make tabs obvious: larger, bold, boxed. */
[data-baseweb="tab-list"] {gap: .5rem;}
button[data-baseweb="tab"] {height: auto; padding: .6rem 1.1rem; border-radius: 6px 6px 0 0; background: rgba(150,160,175,.12);}
button[data-baseweb="tab"] p {font-size: 1.05rem; font-weight: 650;}
button[data-baseweb="tab"][aria-selected="true"] {background: rgba(109,159,196,.22);}
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
    range_hours = pd.to_numeric(raw_hours.str.extract(r"^(\d{1,2}):\d{2}", expand=False), errors="coerce")
    data["Hour of Day"] = numeric_hours.fillna(range_hours)
    data.loc[~data["Hour of Day"].between(0, 23), "Hour of Day"] = float("nan")

    for column in ["Weather Condition", "Road Class"]:
        data[column] = data[column].fillna("Unknown").astype(str).astype("category")

    # Derive weekdays from the date, avoiding inconsistently formatted day labels.
    data["Weekday"] = pd.Categorical(data["Crash Date"].dt.day_name(), categories=DAY_ORDER, ordered=True)
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


def compute_year_bounds(data):
    """First/last crash date per year, plus which years are fully covered.

    With several cities, a year only counts as covered up to the *earliest* city cutoff,
    so a day that one city has and the other doesn't is never treated as a complete day.
    """
    dated = data.dropna(subset=["Crash Date", "Crash Year"])
    per_city = dated.groupby(["Crash Year", "City"], observed=True)["Crash Date"].agg(["min", "max"])
    first, last, complete = {}, {}, []
    for year, rows in per_city.groupby(level=0):
        year = int(year)
        first[year] = rows["min"].max()
        last[year] = rows["max"].min()
        if first[year] <= pd.Timestamp(year, 1, 14) and last[year] >= pd.Timestamp(year, 12, 17):
            complete.append(year)
    return first, last, sorted(complete)


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
        return pd.DataFrame(columns=["lat", "lon", "Severity", "Street", "Date", "r", "g", "b"])

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
    })
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
    st.title("DFW Crash Map")
    st.caption("DALLAS–FORT WORTH / CRASH ANALYTICS")
    st.divider()
    st.subheader("Filters")
    st.selectbox("City", CITY_CHOICES, key="city")
    selected_year = st.selectbox("Crash year", years, key="year")

    year_data = df.loc[df["Crash Year"] == selected_year]
    weather_options = sorted(year_data["Weather Condition"].astype(str).unique().tolist())
    road_options = sorted(year_data["Road Class"].astype(str).unique().tolist())
    period_options = list(TIME_PERIODS)

    # Drop stale selections (e.g. after switching years) before the widgets are created.
    for key, options in [("f_weather", weather_options), ("f_roads", road_options), ("f_periods", period_options)]:
        st.session_state[key] = [v for v in st.session_state.get(key, []) if v in options]

    st.button("Reset filters", on_click=reset_filters, width="stretch")
    st.caption("Leave a filter empty to include all values.")
    selected_weather = st.multiselect("Weather conditions", weather_options, format_func=weather_label,
                                      key="f_weather", placeholder="All conditions")
    selected_roads = st.multiselect("Road types", road_options, format_func=road_label,
                                    key="f_roads", placeholder="All road types")
    selected_periods = st.multiselect("Time of day", period_options,
                                      key="f_periods", placeholder="All times")

    st.divider()
    layout_mode = st.radio("Page layout", ["Single page", "Tabs"], horizontal=True, key="layout",
                           help="Single page shows every section in one scroll. Tabs show one section at a time.")

    st.divider()
    st.caption(f"Source: TxDOT CRIS · {region_name}")
    partial_years = [y for y in years if y not in complete_years]
    for y in partial_years:
        if city_choice == BOTH_LABEL:
            cutoffs = df.loc[df["Crash Year"] == y].groupby("City", observed=True)["Crash Date"].max()
            parts = "; ".join(f"{city} through {fmt_date(day)}" for city, day in cutoffs.items() if pd.notna(day))
            st.caption(f"{y} is a partial year: {parts}.")
        elif y in last_dates:
            st.caption(f"{y} crash records through {fmt_date(last_dates[y])} (partial year).")
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
# Header and overview metrics
# --------------------------------------------------
st.title("DFW Crash Map")
st.caption(f"{region_name.upper()}  /  TRANSPORTATION DATA")
st.write(f"Explore reported crashes across {region_name} by time, roadway, severity, and location.")
st.divider()

st.subheader("Traffic overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Reported crashes", f"{len(filtered):,}")
c2.metric("Crashes with injuries", f"{int(filtered['Injury Crash'].sum()):,}")
c3.metric("Fatal crashes", f"{int(filtered['Fatal Crash'].sum()):,}")
c4.metric("Injury crash share", f"{filtered['Injury Crash'].mean() * 100:.1f}%" if len(filtered) else "—")
st.caption(f"Showing {selected_year} · All charts use the selected sidebar filters unless labeled otherwise.")

if filtered.empty:
    st.warning("No crashes match the current filters. Try removing some filters or choosing another year.")
    st.stop()

SECTION_NAMES = ["Overview", "When crashes happen", "Streets & road types", "Year comparison", "Map"]
if layout_mode == "Tabs":
    st.info("Use the tabs below to switch sections. Each one has its own charts. "
            "Prefer one long page? Change **Page layout** in the sidebar.")
    tab_overview, tab_time, tab_roads, tab_compare, tab_map = st.tabs(SECTION_NAMES)
else:
    sections = []
    for index in range(len(SECTION_NAMES)):
        if index:
            st.divider()
        sections.append(st.container())
    tab_overview, tab_time, tab_roads, tab_compare, tab_map = sections

# --------------------------------------------------
# Overview: insights
# --------------------------------------------------
with tab_overview:
    st.subheader("Crash insights")
    st.caption("Quick observations from the currently selected year and filters")

    valid_hours = filtered["Hour of Day"].dropna()
    peak_hour = int(valid_hours.value_counts().idxmax()) if not valid_hours.empty else None
    peak_count = int((valid_hours == peak_hour).sum()) if peak_hour is not None else 0
    peak_text = hour_label(peak_hour) if peak_hour is not None else "No data"

    wk = weekday_table(filtered, selected_year, first_dates, last_dates)
    wk_valid = wk.dropna(subset=["Per day"])
    if not wk_valid.empty and wk_valid["Per day"].max() > 0:
        top_wk = wk_valid.loc[wk_valid["Per day"].idxmax()]
        weekday_text, weekday_note = top_wk["Weekday"], f"{top_wk['Per day']:.1f} reported crashes per {top_wk['Weekday']}, on average"
    else:
        weekday_text, weekday_note = "No data", ""

    street_counts = filtered[street_col].dropna().value_counts()
    top_road = street_counts.index[0] if not street_counts.empty else "No data"
    top_road_count = int(street_counts.iloc[0]) if not street_counts.empty else 0

    ia, ib, ic = st.columns(3, gap="medium")
    with ia:
        st.metric("Peak crash hour", peak_text)
        st.caption(f"{peak_count:,} reported crashes at this hour")
    with ib:
        st.metric("Busiest weekday (per day)", weekday_text)
        st.caption(weekday_note)
    with ic:
        st.metric("Highest-count roadway", top_road)
        st.caption(f"{top_road_count:,} reported crashes on this roadway")

    st.caption(
        "These are counts within the current filters. They do not account for traffic volume or road length, "
        "so they show where crashes are reported, not where driving is riskiest."
    )

    if city_choice == BOTH_LABEL:
        st.divider()
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
    mode = st.radio("Show", ["Average crashes per day", "Total crashes"], horizontal=True, key="weekday_mode")
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
    fig.update_xaxes(tickmode="array", tickvals=hour_names[::3], ticktext=hour_names[::3], title=None)
    fig.update_yaxes(title=None)
    fig.update_layout(coloraxis_colorbar=dict(title="Crashes", thickness=12))
    show(fig)

    st.divider()
    st.subheader("Severity by hour")
    st.caption(f"Share of reported crashes that involved an injury. Hours with fewer than {MIN_HOUR_SAMPLE} crashes are hidden.")
    sev = filtered.dropna(subset=["Hour of Day"]).assign(Hour=lambda d: d["Hour of Day"].astype(int))
    sev = sev.groupby("Hour").agg(Crashes=("Hour", "size"), Injury=("Injury Crash", "sum")).reindex(range(24), fill_value=0)
    sev["Injury share"] = np.where(sev["Crashes"] >= MIN_HOUR_SAMPLE, sev["Injury"] / sev["Crashes"].replace(0, np.nan) * 100, np.nan)
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
    rank_by = st.selectbox("Rank streets by", ["Total crashes", "Crashes with injuries", "Fatal crashes"], key="street_rank")
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
        fig = px.bar(top.sort_values(rank_column), x=rank_column, y="Street", orientation="h", text=rank_column)
        fig.update_traces(hovertemplate="<b>%{y}</b><br>" + rank_by + ": %{x:,}<extra></extra>",
                          textposition="outside", cliponaxis=False)
        fig.update_layout(yaxis_title=None, xaxis_title=rank_by)
        show(plot_style(fig, 410, horizontal=True, color=metric_color(rank_by)))
        st.dataframe(top[["Street", "Crashes", "Injury crashes", "Fatal crashes"]], hide_index=True, width="stretch")
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
        fig = px.bar(road_counts.sort_values("Crashes"), x="Crashes", y="Road type", orientation="h")
        fig.update_traces(hovertemplate="<b>%{y}</b><br>Crashes: %{x:,}<extra></extra>")
        fig.update_yaxes(title=None)
        show(plot_style(fig, 390, horizontal=True))
    with right:
        st.markdown("**Injury crash share by road type**")
        fig = px.bar(road_counts.sort_values("Injury share"), x="Injury share", y="Road type", orientation="h")
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
        st.info("At least two complete calendar years are needed for a comparison.")
    else:
        st.write("Compare two complete calendar years using the weather, road, and time filters selected in the sidebar.")
        ca, cb = st.columns(2)
        year_a = ca.selectbox("Baseline year", complete_years, index=len(complete_years) - 2, key="cmp_a")
        year_b = cb.selectbox("Comparison year", complete_years, index=len(complete_years) - 1, key="cmp_b")
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
                st.metric(f"{year_b} vs. {year_a} · {compare_metric.lower()}", f"{new_count:,}", f"{delta:+.1f}%", delta_color="off")
                if new_count == old_count:
                    st.caption(f"**Annual insight:** {compare_metric} were unchanged between {year_a} and {year_b} under the selected filters.")
                else:
                    direction = "higher" if new_count > old_count else "lower"
                    st.caption(f"**Annual insight:** {compare_metric} were {abs(delta):.1f}% {direction} in {year_b} than in {year_a} "
                               f"under the selected filters ({old_count:,} → {new_count:,}).")
            else:
                st.caption(f"Percentage change unavailable because the {year_a} baseline is zero.")
            st.caption("Counts reflect reported crashes, not traffic-volume-adjusted risk. Partial years are not offered here.")

# --------------------------------------------------
# Map
# --------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=16)
def build_map(_data, city, year, weather, roads, periods, severity, map_style):
    """Builds the full map HTML once per unique selection, so unrelated reruns reuse it."""
    coords = map_data(_data, city, year, weather, roads, periods, severity)
    if coords.empty:
        return None

    if map_style == "Hotspot Heatmap":
        points = heatmap_cells(coords)
        layer = pdk.Layer("HeatmapLayer", data=points, get_position="[lon, lat]",
                          get_weight="weight", radius_pixels=35, opacity=0.7)
        caption = (f"Representing {len(coords):,} geocoded crash records using {len(points):,} weighted cells. "
                   "Locations are rounded to a geographic grid.")
        tooltip = None
    else:
        points = coords.sample(n=MAX_SCATTER_POINTS, random_state=42) if len(coords) > MAX_SCATTER_POINTS else coords
        layer = pdk.Layer(
            "ScatterplotLayer", data=points, get_position="[lon, lat]",
            get_radius=25, radius_min_pixels=2, get_fill_color="[r, g, b, 170]",
            pickable=True, auto_highlight=True,
        )
        caption = (f"Showing {len(points):,} of {len(coords):,} geocoded crashes. "
                   "Individual locations are sampled only when necessary. Hover a point for details.")
        tooltip = {"html": "<b>{Street}</b><br/>{Date}<br/>{Severity}"}

    view_lat, view_lon, view_zoom = CITY_VIEWS[city]
    deck_kwargs = dict(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=view_lat, longitude=view_lon, zoom=view_zoom, pitch=0),
        map_style="dark",
    )
    if tooltip:  # passing tooltip=None can break the embedded map, so only set it when needed
        deck_kwargs["tooltip"] = tooltip
    deck = pdk.Deck(**deck_kwargs)
    # Streamlit's native pydeck widget keeps its camera across reruns, so a Python-side
    # "reset" can't move it. Embedding the same deck in an iframe lets the Reset button
    # reload the iframe, which creates a fresh map at the starting camera.
    escaped_deck_html = html.escape(deck.to_html(as_string=True, notebook_display=False), quote=True)
    map_embed = f"""
    <!doctype html>
    <html>
    <head>
      <style>
        html, body {{ margin: 0; padding: 0; background: #0e1117; color: #d5dce6;
                     font-family: Arial, sans-serif; }}
        .toolbar {{ display: flex; align-items: center; justify-content: flex-end;
                    height: 43px; padding: 0 4px; }}
        button {{ background: #202938; color: #e7edf6; border: 1px solid #526073;
                  border-radius: 5px; padding: 7px 13px; cursor: pointer; font-size: 13px; }}
        button:hover {{ background: #304057; }}
        iframe {{ display: block; width: 100%; height: 525px; border: 0; }}
      </style>
    </head>
    <body>
      <div class="toolbar">
        <button type="button" onclick="resetView()">Reset map view</button>
      </div>
      <iframe id="crash-map" title="Crash map" srcdoc="{escaped_deck_html}"></iframe>
      <script>
        function resetView() {{
          const frame = document.getElementById('crash-map');
          frame.srcdoc = frame.srcdoc;
        }}
      </script>
    </body>
    </html>
    """
    return map_embed.strip(), caption


@st.fragment
def render_map(data, city, year, weather, roads, periods):
    """Runs as a fragment: changing the map dropdowns reruns only this function, not the whole page."""
    m1, m2 = st.columns(2)
    severity = m1.selectbox("Show crashes", ["All Crashes", "Injury Crashes", "Fatal Crashes"])
    map_style = m2.selectbox("Map visualization", ["Hotspot Heatmap", "Individual Crashes"])

    result = build_map(data, city, year, weather, roads, periods, severity, map_style)
    if result is None:
        st.info("No valid crash coordinates match these filters.")
        return
    map_embed, caption = result
    if hasattr(st, "iframe"):  # st.components.v1.html is deprecated in recent Streamlit releases
        st.iframe(map_embed, height=575)
    else:
        components.html(map_embed, height=575, scrolling=False)
    st.caption(caption)
    if map_style == "Individual Crashes":
        st.caption("Colors: blue = no injury reported · amber = injury · red = fatal.")
    st.caption("Map hotspots represent concentrations of reported crashes, not risk per driver or mile.")


with tab_map:
    st.subheader(f"{region_name} crash map")
    st.write("Explore where crashes were reported. The heatmap uses weighted geographic cells to reduce rendering load.")
    render_map(df, city_choice, selected_year, weather, roads, periods)

st.divider()
st.caption(
    f"Source: Texas Department of Transportation CRIS. Data reflects reported crashes in {region_name}. "
    "Crash concentration is not the same as crash risk per vehicle or mile traveled."
)