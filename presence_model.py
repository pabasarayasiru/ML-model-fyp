# import pandas as pd
# import numpy as np

# # ==============================
# # SETTINGS
# # ==============================
# WINDOW_SECONDS = 30
# STEP_SECONDS = 1
# MIN_ROWS = 10

# # From your Colab output
# THRESHOLD = -50.0

# # In your output:
# # presence RSSI is lower than empty RSSI
# DIRECTION = "lower"

# # Majority voting threshold
# # Example: if >= 60% windows are presence => final presence
# VOTE_RATIO = 0.60


# def create_time_windows(df):
#     df = df.copy()

#     if "Date_Time" in df.columns:
#         df["Date_Time"] = pd.to_datetime(df["Date_Time"], errors="coerce")
#     elif "date_time" in df.columns:
#         df["Date_Time"] = pd.to_datetime(df["date_time"], errors="coerce")
#     else:
#         raise ValueError("Date_Time column not found")

#     if "rssi" not in df.columns:
#         raise ValueError("rssi column not found")

#     df["rssi"] = pd.to_numeric(df["rssi"], errors="coerce")
#     df = df.dropna(subset=["Date_Time", "rssi"])
#     df = df.sort_values("Date_Time").reset_index(drop=True)

#     rows = []

#     start_time = df["Date_Time"].min()
#     last_time = df["Date_Time"].max()
#     current_start = start_time

#     while current_start + pd.Timedelta(seconds=WINDOW_SECONDS) <= last_time:
#         current_end = current_start + pd.Timedelta(seconds=WINDOW_SECONDS)

#         window_df = df[
#             (df["Date_Time"] >= current_start) &
#             (df["Date_Time"] < current_end)
#         ]

#         if len(window_df) < MIN_ROWS:
#             current_start += pd.Timedelta(seconds=STEP_SECONDS)
#             continue

#         rssi_values = window_df["rssi"].values

#         rows.append({
#             "start_time": current_start,
#             "end_time": current_end,
#             "row_count": len(window_df),
#             "mean_rssi": np.mean(rssi_values)
#         })

#         current_start += pd.Timedelta(seconds=STEP_SECONDS)

#     return pd.DataFrame(rows)


# def predict_presence_from_file(csi_file_path):
#     df = pd.read_csv(csi_file_path)

#     windows = create_time_windows(df)

#     if windows.empty:
#         print("No valid RSSI windows")
#         return None

#     if DIRECTION == "lower":
#         windows["prediction"] = np.where(
#             windows["mean_rssi"] < THRESHOLD,
#             "presence",
#             "absence"
#         )
#     else:
#         windows["prediction"] = np.where(
#             windows["mean_rssi"] > THRESHOLD,
#             "presence",
#             "absence"
#         )

#     counts = windows["prediction"].value_counts()

#     presence_count = counts.get("presence", 0)
#     absence_count = counts.get("absence", 0)
#     total_count = len(windows)

#     presence_ratio = presence_count / total_count
#     absence_ratio = absence_count / total_count

#     print("Presence detection window counts:")
#     print(counts)

#     print("Presence ratio:", round(presence_ratio, 3))
#     print("Absence ratio:", round(absence_ratio, 3))

#     if presence_ratio >= VOTE_RATIO:
#         return "presence"
#     else:
#         return "absence"
    






from collections import deque
from datetime import datetime, timedelta
import numpy as np


# ==============================
# SETTINGS
# ==============================
WINDOW_SECONDS = 30
MIN_ROWS = 10

THRESHOLD = -50.0
DIRECTION = "lower"

MIN_VALID_RSSI = -100
MAX_VALID_RSSI = -20

PRESENCE_CONFIRM_COUNT = 5
ABSENCE_CONFIRM_COUNT = 5


class LivePresenceDetector:
    def __init__(self):
        self.rssi_buffer = deque()

        self.presence_streak = 0
        self.absence_streak = 0

        self.final_prediction = "absence"

    def update(self, timestamp, rssi):
        """
        Call this function for every new live CSI RSSI value.
        """

        try:
            rssi = float(rssi)
        except:
            return {
                "status": "ignored",
                "message": "RSSI is not numeric",
                "rssi": rssi
            }

        if rssi < MIN_VALID_RSSI or rssi > MAX_VALID_RSSI:
            return {
                "status": "ignored",
                "message": "RSSI out of valid range",
                "rssi": rssi
            }

        if timestamp is None:
            timestamp = datetime.now()

        # Add new RSSI value
        self.rssi_buffer.append((timestamp, rssi))

        # Keep only last 30 seconds
        cutoff_time = timestamp - timedelta(seconds=WINDOW_SECONDS)

        while self.rssi_buffer and self.rssi_buffer[0][0] < cutoff_time:
            self.rssi_buffer.popleft()

        # Wait until enough rows
        if len(self.rssi_buffer) < MIN_ROWS:
            return {
                "status": "waiting",
                "message": "Not enough RSSI data yet",
                "row_count": len(self.rssi_buffer),
                "final_prediction": self.final_prediction
            }

        # Median RSSI is better for sudden interference drops
        rssi_values = [item[1] for item in self.rssi_buffer]
        median_rssi = np.median(rssi_values)

        # Current prediction
        if DIRECTION == "lower":
            current_prediction = "presence" if median_rssi < THRESHOLD else "absence"
        else:
            current_prediction = "presence" if median_rssi > THRESHOLD else "absence"

        # Consecutive confirmation
        if current_prediction == "presence":
            self.presence_streak += 1
            self.absence_streak = 0
        else:
            self.absence_streak += 1
            self.presence_streak = 0

        # Stable final prediction update
        if self.presence_streak >= PRESENCE_CONFIRM_COUNT:
            self.final_prediction = "presence"

        if self.absence_streak >= ABSENCE_CONFIRM_COUNT:
            self.final_prediction = "absence"

        return {
            "status": "ok",
            "current_prediction": current_prediction,
            "final_prediction": self.final_prediction,
            "median_rssi": round(median_rssi, 2),
            "row_count": len(self.rssi_buffer),
            "presence_streak": self.presence_streak,
            "absence_streak": self.absence_streak,
            "threshold": THRESHOLD
        }


# ==============================
# TEST EXAMPLE
# ==============================
if __name__ == "__main__":
    detector = LivePresenceDetector()

    sample_rssi_values = [
        -45, -46, -45, -44, -46,
        -80,
        -45, -46, -44, -45,
        -55, -56, -57, -55, -56,
        -57, -58, -56, -55, -57
    ]

    for rssi in sample_rssi_values:
        result = detector.update(datetime.now(), rssi)
        print(result)