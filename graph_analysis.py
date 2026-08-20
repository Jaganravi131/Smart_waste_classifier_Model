"""Simple graph analysis utilities for detection logs.

This module provides GraphAnalyzer which logs prediction records to a CSV
and helpers to read the CSV into pandas for plotting.
"""
from typing import Optional, Dict, Any
import csv
import os
from datetime import datetime
import pandas as pd


class GraphAnalyzer:
    """Log detection predictions to a CSV and provide helpers for analysis.

    CSV columns:
      timestamp (ISO UTC), class, confidence, x, y, width, height, source

    Methods:
      log_predictions(predictions, timestamp): append rows for each detection
      load_dataframe(): returns a pandas.DataFrame with parsed timestamp
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._ensure_csv()

    def _ensure_csv(self):
        if not os.path.exists(self.csv_path):
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            with open(self.csv_path, "w", newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "class", "confidence", "x", "y", "width", "height", "source"])

    def log_predictions(self, predictions: Dict[str, Any], timestamp: Optional[datetime] = None, source: Optional[str] = None):
        """Append predictions dict to CSV.

        predictions expected format: {"predictions": [{"class":..., "confidence":..., "x":..., "y":..., "width":..., "height":...}, ...]}
        If predictions is None or has no predictions, a single row with empty class is written to mark the frame.
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        rows = []
        preds = predictions.get("predictions") if isinstance(predictions, dict) else None
        if preds:
            for p in preds:
                rows.append([
                    timestamp.isoformat(),
                    p.get("class"),
                    float(p.get("confidence", 0.0)),
                    p.get("x"),
                    p.get("y"),
                    p.get("width"),
                    p.get("height"),
                    source,
                ])
        else:
            # mark an empty frame (no detection)
            rows.append([timestamp.isoformat(), None, None, None, None, None, None, source])

        with open(self.csv_path, "a", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    def load_dataframe(self) -> pd.DataFrame:
        """Load the CSV into a DataFrame and parse timestamps."""
        df = pd.read_csv(self.csv_path, parse_dates=["timestamp"]) 
        return df


def aggregate_counts_by_class(df: pd.DataFrame):
    """Return a series with counts of detections per class."""
    return df[~df["class"].isna()].groupby("class").size().sort_values(ascending=False)


def moving_avg_confidence(df: pd.DataFrame, class_name: str, window: int = 10):
    """Return a DataFrame with rolling average confidence for a given class."""
    sel = df[df["class"] == class_name].set_index("timestamp").sort_index()
    sel = sel["confidence"].dropna().astype(float)
    return sel.rolling(window=window).mean()
