import asyncio
import websockets
import json
from datetime import datetime, timezone
import pandas as pd

from heart_rate_model import (
    get_heart_rate_windows,
    predict_heart_rate_from_window
)

from respiration_model import (
    get_respiration_windows,
    predict_respiration_rate_from_window
)

from posture_model import predict_posture_from_file

# NEW live presence detector
from presence_model import LivePresenceDetector


URI = "wss://health-app-wifi-csi-monitoring.onrender.com"
CSI_FILE_PATH = "csi/csi_data_2026-05-04_15-50-17_pramod_right_01.csv"


def load_rssi_data(file_path):
    df = pd.read_csv(file_path)

    if "rssi" not in df.columns:
        raise ValueError("rssi column not found in CSV file")

    if "Date_Time" in df.columns:
        df["presence_time"] = pd.to_datetime(df["Date_Time"], errors="coerce")
    elif "date_time" in df.columns:
        df["presence_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    elif "Date_time" in df.columns:
        df["presence_time"] = pd.to_datetime(df["Date_time"], errors="coerce")
    else:
        df["presence_time"] = None

    return df[["presence_time", "rssi"]].dropna(subset=["rssi"]).reset_index(drop=True)


async def send_data():
    heart_windows = get_heart_rate_windows(CSI_FILE_PATH)
    resp_windows = get_respiration_windows(CSI_FILE_PATH)

    posture = predict_posture_from_file(CSI_FILE_PATH)

    if posture is None:
        print("Posture prediction failed")
        return

    posture = posture.lower()
    print("Final predicted posture:", posture)
    print("Total HR windows:", len(heart_windows))
    print("Total RR windows:", len(resp_windows))

    # NEW presence detector setup
    presence_detector = LivePresenceDetector()
    rssi_data = load_rssi_data(CSI_FILE_PATH)
    rssi_index = 0

    total_windows = min(len(heart_windows), len(resp_windows))

    if total_windows == 0:
        print("No valid windows found")
        return

    index = 0
    reconnect_delay = 5

    while True:
        try:
            async with websockets.connect(
                URI,
                ping_interval=30,
                ping_timeout=60,
                close_timeout=10,
                open_timeout=60,
                max_size=None
            ) as websocket:

                print("Connected to backend")
                reconnect_delay = 5

                while True:
                    if index >= total_windows:
                        print("Finished all CSI windows. Replaying again...")
                        index = 0

                    if rssi_index >= len(rssi_data):
                        print("Finished RSSI data. Replaying again...")
                        rssi_index = 0

                    heart_window = heart_windows[index]
                    resp_window = resp_windows[index]

                    try:
                        heart_rate = predict_heart_rate_from_window(heart_window)
                    except Exception as e:
                        print("Heart rate prediction error:", e)
                        index += 1
                        continue

                    try:
                        respiration_rate = predict_respiration_rate_from_window(resp_window)
                    except Exception as e:
                        print("Respiration rate prediction error:", e)
                        index += 1
                        continue

                    # NEW live presence update
                    rssi_row = rssi_data.iloc[rssi_index]
                    timestamp = rssi_row["presence_time"]

                    if pd.isna(timestamp):
                        timestamp = datetime.now()

                    presence_result = presence_detector.update(timestamp, rssi_row["rssi"])
                    presence = presence_result.get("final_prediction", "absence")

                    print(
                        f"HR: {heart_rate} | "
                        f"RR: {respiration_rate} | "
                        f"Posture: {posture} | "
                        f"Presence: {presence} | "
                        f"Presence details: {presence_result}"
                    )

                    if heart_rate is None or respiration_rate is None or posture is None or presence is None:
                        print("Skipping invalid prediction")
                        index += 1
                        rssi_index += 1
                        continue

                    message = {
                        "type": "health_data",
                        "payload": {
                            "heart_rate": round(float(heart_rate), 2),
                            "respiration_rate": round(float(respiration_rate), 2),
                            "posture": posture,
                            "presence": presence,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                    }

                    await websocket.send(json.dumps(message))
                    print(f"Sent window {index}:", message)

                    index += 1
                    rssi_index += 1

                    await asyncio.sleep(2)

        except websockets.exceptions.ConnectionClosed as e:
            print("WebSocket closed")
            print("Code:", e.code)
            print("Reason:", e.reason)

        except Exception as e:
            print("Disconnected:", e)

        print(f"Reconnecting in {reconnect_delay} seconds...")
        await asyncio.sleep(reconnect_delay)

        reconnect_delay = min(reconnect_delay * 2, 60)


asyncio.run(send_data())