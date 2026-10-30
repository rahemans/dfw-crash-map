<h1 align="center">DFW Crash Map</h1>

<p align="center">Explore where, when, and under what conditions traffic crashes happen across Dallas and Fort Worth.</p>

<p align="center">
  <a href="https://dfw-crash-map-ujqh4w6674qx5pszdlthdb.streamlit.app/">Live dashboard</a>
  · <a href="https://github.com/rahemans/dfw-crash-map">Source code</a>
</p>

![DFW Crash Map — crash locations in downtown Dallas](assets/crash-mapview.png)

## Why I built it

Driving around DFW can be an experience, especially during rush hour. I wanted to go beyond complaining about traffic and see what the crash data actually showed.

So I built DFW Crash Map using reported crash records from the Texas Department of Transportation (TxDOT).

## What you can explore

- **Crash locations:** Switch between individual crash points and a hotspot heatmap. Filter for crashes involving injuries or fatalities, and inspect reported injury and fatality counts.
- **Filters:** Explore Dallas, Fort Worth, or both by year, weather, road type, and time of day.
- **Time patterns:** Compare crashes by month, hour, weekday, and hour-by-weekday combinations.
- **Roadway patterns:** Explore the streets with the most reported crashes, injury crashes, or fatal crashes, and export rankings as CSV.
- **City and year comparisons:** Compare reported crash patterns across Dallas and Fort Worth and avoid treating partial-year exports as complete years.

![DFW Crash Map — crash trends and charts](assets/crash-trends.png)

## Built with

**Python** · **Streamlit** · **pandas** · **NumPy** · **Plotly** · **PyDeck**

The app cleans TxDOT CRIS exports and uses optional Parquet caching to make repeat loads faster. Map points are sampled when needed for performance, while fatal-crash points are retained.

## Run locally

```bash
git clone https://github.com/rahemans/dfw-crash-map.git
cd dfw-crash-map
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

You'll need TxDOT CRIS CSV exports in the `Data/` folder, preserving the original field names:

```text
Data/
├── dallas_crashes_2024_2026.csv
└── forthworth_crashes_2024_2026.csv
```

The `forthworth` spelling matches the filename expected by the app. Either city can load independently; both files are needed for combined views. On Windows, activate the virtual environment with `.venv\Scripts\Activate.ps1` instead.

### Tests

```bash
python3 -m pip install pytest
python3 -m pytest -q tests/test_crash_logic.py
```

## About the data

**Source:** Texas Department of Transportation, Crash Records Information System (CRIS), configured for 2024–2026 exports. The latest year may be incomplete.

The map shows records with usable coordinates; other charts may include records without map locations. Injury and fatal crash categories can overlap. Individual points may be sampled for performance, but fatal-crash points are retained.

**Reported crash counts are not traffic-adjusted risk.** A road with more recorded crashes is not necessarily more dangerous per driver, mile, or trip. The dashboard does not predict crashes or determine fault.

---

<sub>Independent student data science project. Not affiliated with or endorsed by TxDOT.</sub>
