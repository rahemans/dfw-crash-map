# DFW Crash Map 💥

**Explore where and when traffic crashes are reported across Dallas and Fort Worth.**

DFW Crash Map is an interactive data dashboard built with Python and Streamlit. It turns Texas Department of Transportation (TxDOT) Crash Records Information System (CRIS) exports into accessible visualizations of crash locations, timing, roadway types, and injury severity.

> **Scope:** This project explores historical *reported* crashes. It does not predict crashes, determine fault, or measure the probability of a crash on a given road.

## What you can explore

- **Dallas, Fort Worth, or both:** Switch between city-level views and compare reported crash totals and injury-crash shares.
- **When crashes happen:** Examine monthly trends, hourly patterns, weekday averages, and an hour-by-weekday heatmap.
- **Crash severity:** Explore injury and fatal crash counts and how injury-crash share varies by hour and roadway type.
- **Streets and road types:** Rank roadways by total, injury, or fatal crashes and export the full street ranking as CSV.
- **Interactive map:** Explore geographic hotspots or individual geocoded crash records, with filters for city, year, weather, road class, and time of day.
- **Year-over-year comparisons:** Compare available complete years without treating an incomplete year as a full one.
- **Shareable views:** The URL preserves the selected city, year, and main filters.

## Preview

*Screenshot coming soon — add an image of the running dashboard here after deployment.*

## Tech stack

Python · Streamlit · pandas · NumPy · Plotly · PyDeck · PyArrow (for optional Parquet caching)

## Run locally

**1. Clone the repository**

```bash
git clone https://github.com/rahemans/dfw-crash-map.git
cd dfw-crash-map
```

**2. Create and activate a virtual environment (recommended)**

```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
# Windows PowerShell: .venv\Scripts\Activate.ps1
```

**3. Install dependencies**

```bash
python -m pip install -r requirements.txt
```

**4. Add the source data**

Place the TxDOT CRIS CSV exports in a `Data/` folder, using these filenames:

```text
Data/
├── dallas_crashes_2024_2026.csv
└── forthworth_crashes_2024_2026.csv
```

The `forthworth` spelling is intentional: it matches the filename expected by `app.py`. The app can start with just one of the two files, but both are needed for combined-city comparisons. The exports should retain their original TxDOT column headers, including crash date, coordinates, severity counts, and roadway information.

**5. Start the app**

```bash
python -m streamlit run app.py
```

Open the local address printed in your terminal (typically `http://localhost:8501`).

## Data and methodology

**Source:** Texas Department of Transportation, Crash Records Information System (CRIS). The dashboard currently supports the 2024–2026 export files listed above. Actual coverage depends on the contents and cutoff dates of those exports; recent years may be incomplete.

- The app cleans date, hour, severity, street, and location fields before analysis.
- Maps include only records with usable coordinates inside the configured Dallas–Fort Worth bounding area; charts may include records without coordinates.
- The hotspot view aggregates crash locations into geographic cells. The individual-point view samples records when there are too many to render comfortably.
- Street rankings count reported crashes associated with a street name, **not** crash rates adjusted for traffic volume, road length, or number of trips.
- City comparisons use raw reported counts and injury shares; differences in exposure and reporting mean they should not be interpreted as a definitive safety ranking.

## Project structure

```text
dfw-crash-map/
├── app.py                 # Streamlit dashboard and data processing
├── requirements.txt       # Python dependencies
├── .gitignore             # Local environment and generated cache exclusions
├── Data/                  # TxDOT CRIS source CSVs (if redistribution is permitted)
└── README.md
```

Generated `.parquet` files are local caches and do not need to be committed. Before publishing raw CSVs, verify their size and that public redistribution is permitted by the data provider.

## Future improvements

Potential extensions include traffic-volume-normalized rates, additional cities, longer historical coverage, and more detailed location-level analysis.

---

*Independent student data science project. Not affiliated with or endorsed by TxDOT.*
