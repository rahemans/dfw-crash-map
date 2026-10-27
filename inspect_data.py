
import pandas as pd

# Load Dallas crash records
df = pd.read_csv(
    "Data/dallas_crashes_2024_2026.csv",
    skiprows=11,
    low_memory=False
)

# Show basic information
print("Total crashes:", len(df))
print("Total columns:", len(df.columns))

print("\nCrashes by year:")
print(df["Crash Year"].value_counts().sort_index())

print("\nColumn names:")
print(df.columns.tolist())
