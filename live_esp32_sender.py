import asyncio
import websockets
import json
import os
import glob
import random
import pandas as pd
from datetime import datetime, timezone

from heart_rate_model import (
    get_heart_rate_windows,
    predict_heart_rate_from_window
)

from respiration_model import (
    get_respiration_windows,
    predict_respiration_rate_from_window
)

URI = "wss://health-app-wifi-csi-monitoring.onrender.com"

CSI_FOLDER = r"C:\Users\ASUS\Documents\esp-csi\examples\get-started\tools"

TEMP_FIXED_FILE = r"csi\live_fixed_csi_data.csv"

SEND_INTERVAL_SECONDS = 2
MIN_ROWS = 300

CORRECT_COLUMNS = [
    "date_time", "type", "id", "mac", "rssi", "rate", "sig_mode", "mcs",
    "bandwidth", "smoothing", "not_sounding", "aggregation", "stbc",
    "fec_coding", "sgi", "noise_floor", "ampdu_cnt", "channel",
    "secondary_channel", "local_timestamp", "ant", "sig_len", "rx_state",
    "len", "first_word", "data"
]


def get_latest_csv_file():
    csv_files = glob.glob(os.path.join(CSI_FOLDER, "*.csv"))

    if not csv_files:
        return None

    return max(csv_files, key=os.path.getmtime)


def fix_csv_header(source_file):
    os.makedirs("csi", exist_ok=True)

    df = pd.read_csv(
        source_file,
        header=0,
        names=CORRECT_COLUMNS,
        on_bad_lines="skip"
    )

    df.to_csv(TEMP_FIXED_FILE, index=False)

    return TEMP_FIXED_FILE, len(df)


async def predict_and_send(websocket, csv_file):
    heart_windows = get_heart_rate_windows(csv_file)
    resp_windows = get_respiration_windows(csv_file)

    if len(heart_windows) == 0 or len(resp_windows) == 0:
        print("Not enough valid windows yet")
        return

    heart_window = heart_windows[-1]
    resp_window = resp_windows[-1]

    heart_rate = predict_heart_rate_from_window(heart_window)
    respiration_rate = predict_respiration_rate_from_window(resp_window)

    if heart_rate is None or respiration_rate is None:
        print("Invalid prediction")
        return

    message = {
        "type": "health_data",
        "payload": {
            "heart_rate": round(float(heart_rate), 2),
            "respiration_rate": round(float(respiration_rate), 2),
            "posture": random.choice(["supine", "prone", "left", "right"]),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    }

    await websocket.send(json.dumps(message))
    print("Sent:", message)


async def main():
    print("Watching CSI folder:")
    print(CSI_FOLDER)

    async with websockets.connect(
        URI,
        ping_interval=30,
        ping_timeout=60,
        close_timeout=10,
        open_timeout=60,
        max_size=None
    ) as websocket:

        print("Connected to backend")

        while True:
            source_csv = get_latest_csv_file()

            if source_csv is None:
                print("No CSV file found yet...")
                await asyncio.sleep(2)
                continue

            try:
                fixed_csv, rows = fix_csv_header(source_csv)

                print("Source CSV:", source_csv)
                print("Fixed CSV:", fixed_csv)
                print("Rows:", rows)

                if rows >= MIN_ROWS:
                    await predict_and_send(websocket, fixed_csv)
                else:
                    print("Waiting for more CSI rows...")

            except Exception as e:
                print("Error:", e)

            await asyncio.sleep(SEND_INTERVAL_SECONDS)


if __name__ == "__main__":
    asyncio.run(main())