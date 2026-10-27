"""Unit tests for DFW Crash Map's real data-processing functions.

Runs only the selected functions from app.py, avoiding launching Streamlit.
"""
import ast
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

APP = Path(__file__).resolve().parents[1] / 'app.py'
SOURCE = ast.parse(APP.read_text(encoding='utf-8'))
FUNCTIONS = {'clean', 'apply_filters', 'compute_year_bounds', 'weekday_table', 'find_header_row', 'format_street_name'}
CONSTANTS = {'DAY_ORDER', 'TIME_PERIODS', 'FALLBACK_HEADER_ROW', 'KNOWN_STREET_NAMES', 'HIGHWAY_PREFIXES'}
import re
ns = {'pd': pd, 'np': np, 're': re}
for node in SOURCE.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in CONSTANTS:
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(APP), 'exec'), ns)
# Regexes are constructed independently from the exact patterns in app.py.
ns['HIGHWAY_PATTERN'] = re.compile(r'(IH|US|SH|SL|SS|FM|RM|BU)(0*[0-9]+)([A-Z]?)')
ns['OTHER_CODE_PATTERN'] = re.compile(r'[A-Z]{2}0*[0-9]{2,}[A-Z]?')
for node in SOURCE.body:
    if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS:
        node.decorator_list = []
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(APP), 'exec'), ns)


def sample():
    return pd.DataFrame({
        'Crash Date': ['2025-01-01', '2025-01-02', '2025-01-03', '2026-01-01'],
        'Crash Year': [2025, 2025, 2025, 2026],
        'Crash Month': [1, 1, 1, 1],
        'Crash Total Injury Count': [0, 2, 1, 0],
        'Crash Death Count': [0, 0, 1, 0],
        'Latitude': [32.8] * 4,
        'Longitude': [-96.8] * 4,
        'Hour of Day': ['8', '17:00', '22', '3'],
        'Weather Condition': ['Clear', 'Rain', 'Clear', 'Rain'],
        'Road Class': ['Interstate', 'Local', 'Interstate', 'Local'],
        'Street Name': ['IH0035E', 'MAIN ST', 'IH0035E', None],
        'City': ['Dallas', 'Fort Worth', 'Dallas', 'Fort Worth'],
    })


class CrashLogicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = ns['clean'](sample())

    def test_injury_and_fatal_counts(self):
        d = self.data
        self.assertEqual(len(d), 4)
        self.assertEqual(int(d['Injury Crash'].sum()), 2)
        self.assertEqual(int(d['Fatal Crash'].sum()), 1)
        self.assertAlmostEqual(d['Injury Crash'].mean() * 100, 50)
        self.assertTrue(bool(d.iloc[2]['Injury Crash']) and bool(d.iloc[2]['Fatal Crash']))

    def test_year_filter(self):
        self.assertEqual(len(ns['apply_filters'](self.data, 2025, (), (), ())), 3)

    def test_weather_and_road_filters(self):
        result = ns['apply_filters'](self.data, 2025, ('Clear',), ('Interstate',), ())
        self.assertEqual(len(result), 2)

    def test_time_period_filter(self):
        result = ns['apply_filters'](self.data, 2025, (), (), ('Evening Rush (4 PM–7 PM)',))
        self.assertEqual(len(result), 1)
        self.assertEqual(int(result.iloc[0]['Hour of Day']), 17)

    def test_year_comparison_filter_ignores_year(self):
        result = ns['apply_filters'](self.data, None, (), (), (), include_year=False)
        self.assertEqual(len(result), 4)

    def test_hour_parsing_and_street_labels(self):
        self.assertEqual(self.data['Hour of Day'].tolist(), [8, 17, 22, 3])
        self.assertEqual(self.data.iloc[0]['Street Display'], 'I-35E')
        self.assertTrue(pd.isna(self.data.iloc[3]['Street Display']))

    def test_weekday_average_full_year(self):
        data = self.data.loc[self.data['Crash Year'] == 2025]
        table = ns['weekday_table'](data, 2025, {}, {2025: pd.Timestamp('2025-12-31')})
        wed = table.set_index('Weekday').loc['Wednesday']
        self.assertEqual(int(wed['Crashes']), 1)
        self.assertEqual(int(wed['Days in period']), 53)
        self.assertAlmostEqual(wed['Per day'], 1 / 53)

    def test_year_bounds_marks_partial_year(self):
        first, last, complete = ns['compute_year_bounds'](self.data)
        self.assertEqual(last[2026], pd.Timestamp('2026-01-01'))
        self.assertNotIn(2026, complete)
        self.assertNotIn(2025, complete)


if __name__ == '__main__':
    unittest.main()
