import pandas as pd
import numpy as np

# ==============================
# SETTINGS
# ==============================
WINDOW_SECONDS = 30
STEP_SECONDS = 1
MIN_ROWS = 10

# From your Colab output
THRESHOLD = -50.0

# In your output:
# presence RSSI is lower than empty RSSI
DIRECTION = "lower"

# Majority voting threshold
# Example: if >= 60% windows are presence => final presence
VOTE_RATIO = 0.60


def create_time_windows(df):
    df = df.copy()

    if "Date_Time" in df.columns:
        df["Date_Time"] = pd.to_datetime(df["Date_Time"], errors="coerce")
    elif "date_time" in df.columns:
        df["Date_Time"] = pd.to_datetime(df["date_time"], errors="coerce")
    else:
        raise ValueError("Date_Time column not found")

    if "rssi" not in df.columns:
        raise ValueError("rssi column not found")

    df["rssi"] = pd.to_numeric(df["rssi"], errors="coerce")
    df = df.dropna(subset=["Date_Time", "rssi"])
    df = df.sort_values("Date_Time").reset_index(drop=True)

    rows = []

    start_time = df["Date_Time"].min()
    last_time = df["Date_Time"].max()
    current_start = start_time

    while current_start + pd.Timedelta(seconds=WINDOW_SECONDS) <= last_time:
        current_end = current_start + pd.Timedelta(seconds=WINDOW_SECONDS)

        window_df = df[
            (df["Date_Time"] >= current_start) &
            (df["Date_Time"] < current_end)
        ]

        if len(window_df) < MIN_ROWS:
            current_start += pd.Timedelta(seconds=STEP_SECONDS)
            continue

        rssi_values = window_df["rssi"].values

        rows.append({
            "start_time": current_start,
            "end_time": current_end,
            "row_count": len(window_df),
            "mean_rssi": np.mean(rssi_values)
        })

        current_start += pd.Timedelta(seconds=STEP_SECONDS)

    return pd.DataFrame(rows)


def predict_presence_from_file(csi_file_path):
    df = pd.read_csv(csi_file_path)

    windows = create_time_windows(df)

    if windows.empty:
        print("No valid RSSI windows")
        return None

    if DIRECTION == "lower":
        windows["prediction"] = np.where(
            windows["mean_rssi"] < THRESHOLD,
            "presence",
            "absence"
        )
    else:
        windows["prediction"] = np.where(
            windows["mean_rssi"] > THRESHOLD,
            "presence",
            "absence"
        )

    counts = windows["prediction"].value_counts()

    presence_count = counts.get("presence", 0)
    absence_count = counts.get("absence", 0)
    total_count = len(windows)

    presence_ratio = presence_count / total_count
    absence_ratio = absence_count / total_count

    print("Presence detection window counts:")
    print(counts)

    print("Presence ratio:", round(presence_ratio, 3))
    print("Absence ratio:", round(absence_ratio, 3))

    if presence_ratio >= VOTE_RATIO:
        return "presence"
    else:
        return "absence"