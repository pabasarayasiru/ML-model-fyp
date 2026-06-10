# import asyncio
# import websockets
# import json
# import random

# from heart_rate_model import (
#     get_heart_rate_windows,
#     predict_heart_rate_from_window
# )

# from respiration_model import predict_respiration_rate


# URI = "wss://health-app-wifi-csi-monitoring.onrender.com"

# CSI_FILE_PATH = "csi/csi_data_2026-01-14_16-57-17_kalpana_front_01.csv"


# async def send_data():

#     windows = get_heart_rate_windows(CSI_FILE_PATH)

#     print("Total CSI windows:", len(windows))

#     index = 0

#     while True:

#         try:
#             async with websockets.connect(
#                 URI,
#                 ping_interval=20,
#                 ping_timeout=30,
#                 close_timeout=10
#             ) as websocket:

#                 print("Connected to backend")

#                 while True:

#                     if index >= len(windows):
#                         print("Finished all CSI windows. Replaying again...")
#                         index = 0

#                     window = windows[index]

#                     heart_rate = predict_heart_rate_from_window(window)

#                     respiration_rate = predict_respiration_rate(CSI_FILE_PATH)

#                     if heart_rate is None:
#                         print("Skipping invalid heart rate window")
#                         index += 1
#                         continue

#                     if respiration_rate is None:
#                         print("Skipping invalid respiration rate")
#                         index += 1
#                         continue

#                     message = {
#                         "type": "health_data",
#                         "payload": {
#                             "heart_rate": round(float(heart_rate), 2),
#                             "respiration_rate": round(float(respiration_rate), 2),
#                             "posture": random.choice(
#                                 ["supine", "prone", "left", "right"]
#                             )
#                         }
#                     }

#                     await websocket.send(json.dumps(message))

#                     print(f"Sent window {index}:", message)

#                     index += 1

#                     await asyncio.sleep(1)

#         except Exception as e:

#             print("Disconnected:", e)
#             print("Reconnecting in 1 second...")

#             await asyncio.sleep(1)


# asyncio.run(send_data())



import asyncio
import websockets
import json
from datetime import datetime, timezone

from heart_rate_model import (
    get_heart_rate_windows,
    predict_heart_rate_from_window
)

from respiration_model import (
    get_respiration_windows,
    predict_respiration_rate_from_window
)

from posture_model import predict_posture_from_file


URI = "wss://health-app-wifi-csi-monitoring.onrender.com"
CSI_FILE_PATH = "csi/csi_data_2026-05-04_15-50-17_pramod_right_01.csv"


async def send_data():
    heart_windows = get_heart_rate_windows(CSI_FILE_PATH)
    resp_windows = get_respiration_windows(CSI_FILE_PATH)
    # Posture prediction - using new posture_model.py
    posture = predict_posture_from_file(CSI_FILE_PATH)

    if posture is None:
        print("Posture prediction failed")
        return

    posture = posture.lower()
    print("Final predicted posture:", posture)    
    print("Total HR windows:", len(heart_windows))
    print("Total RR windows:", len(resp_windows))

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
                    
                    print(
                        f"HR: {heart_rate} | "
                        f"RR: {respiration_rate} | "
                        f"Posture: {posture}"
                    )

                    if heart_rate is None or respiration_rate is None or posture is None:
                        print("Skipping invalid prediction")
                        index += 1
                        continue

                    message = {
                        "type": "health_data",
                        "payload": {
                            "heart_rate": round(float(heart_rate), 2),
                            "respiration_rate": round(float(respiration_rate), 2),
                            "posture": posture ,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                    }

                    await websocket.send(json.dumps(message))
                    print(f"Sent window {index}:", message)

                    index += 1

                    # For overlapping windows, 1 or 2 seconds is okay
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