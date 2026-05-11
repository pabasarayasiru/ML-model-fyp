import asyncio
import websockets
import json
import random

from heart_rate_model import (
    get_heart_rate_windows,
    predict_heart_rate_from_window
)

from respiration_model import predict_respiration_rate


URI = "wss://health-app-wifi-csi-monitoring.onrender.com"

CSI_FILE_PATH = "csi_data_2026-01-14_16-57-17_kalpana_front_01.csv"


def custom_round(value):
    return int(value) + (1 if value - int(value) >= 0.5 else 0)


async def send_data():

    windows = get_heart_rate_windows(CSI_FILE_PATH)

    print("Total CSI windows:", len(windows))

    index = 0

    while True:

        try:
            async with websockets.connect(
                URI,
                ping_interval=20,
                ping_timeout=30,
                close_timeout=10
            ) as websocket:

                print("Connected to backend")

                while True:

                    if index >= len(windows):
                        print("Finished all CSI windows. Replaying again...")
                        index = 0

                    window = windows[index]

                    heart_rate = predict_heart_rate_from_window(window)

                    # RR model stays same: it uses latest 30 seconds internally
                    respiration_rate = predict_respiration_rate(CSI_FILE_PATH)

                    if heart_rate is None:
                        print("Skipping invalid heart rate window")
                        index += 1
                        continue

                    if respiration_rate is None:
                        print("Skipping invalid respiration rate")
                        index += 1
                        continue

                    message = {
                        "type": "health_data",
                        "payload": {
                            "heart_rate": custom_round(heart_rate),
                            "respiration_rate": custom_round(respiration_rate),
                            "posture": random.choice(
                                ["supine", "prone", "left", "right"]
                            )
                        }
                    }

                    await websocket.send(json.dumps(message))

                    print(f"Sent window {index}:", message)

                    index += 1

                    await asyncio.sleep(1)

        except Exception as e:

            print("Disconnected:", e)
            print("Reconnecting in 1 second...")

            await asyncio.sleep(1)


asyncio.run(send_data())