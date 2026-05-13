# kalpana


# import joblib
# import numpy as np
# import pandas as pd

# from scipy.signal import butter, filtfilt, welch
# from scipy.stats import skew, kurtosis


# # ==========================================
# # LOAD RR MODEL
# # ==========================================
# model = joblib.load("rr_model.pkl")
# scaler = joblib.load("scaler.pkl")

# print("RR model loaded")


# WINDOW_SECONDS = 30


# # ==========================================
# # CSI PREPROCESSING FUNCTIONS
# # ==========================================
# def ProcessCSI(data):

#     AmpCSI = np.zeros((len(data), 64))

#     for i in range(len(data)):

#         parts = data.loc[:, "data"].iloc[i].split(",")

#         parts[0] = parts[0].strip("[")
#         parts[-1] = parts[-1].strip("]")

#         ImCSI = np.array(parts[::2], dtype=np.int64)
#         ReCSI = np.array(parts[1::2], dtype=np.int64)

#         AmpCSI[i][:] = np.sqrt(ImCSI**2 + ReCSI**2)

#     return np.concatenate(
#         (
#             AmpCSI[:, 4:11],
#             AmpCSI[:, 12:25],
#             AmpCSI[:, 26:32],
#             AmpCSI[:, 33:39],
#             AmpCSI[:, 40:53],
#             AmpCSI[:, 54:61],
#         ),
#         axis=1,
#     )


# def remove_dc_offset(CSI_data):
#     return CSI_data - np.mean(CSI_data, axis=0)


# def bandpass_filter(data, fs, lowcut, highcut, order=4):

#     nyq = 0.5 * fs

#     b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")

#     return filtfilt(b, a, data)


# # ==========================================
# # FEATURE EXTRACTION
# # ==========================================
# def extract_features(signal, fs):

#     freqs, psd = welch(signal, fs, nperseg=min(512, len(signal)))

#     peak_freq = freqs[np.argmax(psd)]

#     resp_band = (freqs >= 0.1) & (freqs <= 0.6)

#     band_power = np.sum(psd[resp_band])

#     spectral_centroid = np.sum(freqs * psd) / np.sum(psd)

#     return [
#         np.mean(signal),
#         np.std(signal),
#         skew(signal),
#         kurtosis(signal),
#         np.ptp(signal),
#         peak_freq,
#         band_power,
#         spectral_centroid,
#     ]


# # ==========================================
# # RR PREDICTION
# # ==========================================
# def predict_respiration_rate(csi_file_path):

#     data = pd.read_csv(csi_file_path)

#     data["date_time"] = pd.to_datetime(data["date_time"])

#     latest_time = data["date_time"].iloc[-1]

#     window_start = latest_time - pd.Timedelta(seconds=WINDOW_SECONDS)

#     window_data = data[data["date_time"] >= window_start]

#     if len(window_data) < 50:
#         return None

#     data_sub = ProcessCSI(window_data)

#     data_sub = remove_dc_offset(data_sub)

#     signal = np.mean(data_sub, axis=1)

#     time_range = (
#         window_data["date_time"].iloc[-1]
#         - window_data["date_time"].iloc[0]
#     ).total_seconds()

#     if time_range <= 0:
#         return None

#     fs = len(signal) / time_range

#     signal = bandpass_filter(signal, fs, 0.1, 0.5)

#     features = extract_features(signal, fs)

#     features_scaled = scaler.transform([features])

#     rr = model.predict(features_scaled)[0]

#     return round(float(rr), 2)









# update pabasara

import joblib
import numpy as np
import pandas as pd

from scipy.signal import butter, filtfilt, welch
from scipy.stats import skew, kurtosis


model = joblib.load("rr_model.pkl")
scaler = joblib.load("scaler.pkl")

print("RR model loaded")

WINDOW_SECONDS = 30
STEP_SECONDS = 1


def ProcessCSI(data):
    AmpCSI = np.zeros((len(data), 64))

    for i in range(len(data)):
        parts = data.loc[:, "data"].iloc[i].split(",")

        parts[0] = parts[0].strip("[")
        parts[-1] = parts[-1].strip("]")

        ImCSI = np.array(parts[::2], dtype=np.int64)
        ReCSI = np.array(parts[1::2], dtype=np.int64)

        AmpCSI[i][:] = np.sqrt(ImCSI**2 + ReCSI**2)

    return np.concatenate(
        (
            AmpCSI[:, 4:11],
            AmpCSI[:, 12:25],
            AmpCSI[:, 26:32],
            AmpCSI[:, 33:39],
            AmpCSI[:, 40:53],
            AmpCSI[:, 54:61],
        ),
        axis=1,
    )


def remove_dc_offset(CSI_data):
    return CSI_data - np.mean(CSI_data, axis=0)


def bandpass_filter(data, fs, lowcut, highcut, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, data)


def extract_features(signal, fs):
    freqs, psd = welch(signal, fs, nperseg=min(512, len(signal)))

    peak_freq = freqs[np.argmax(psd)]
    resp_band = (freqs >= 0.1) & (freqs <= 0.6)

    band_power = np.sum(psd[resp_band])
    spectral_centroid = np.sum(freqs * psd) / np.sum(psd)

    return [
        np.mean(signal),
        np.std(signal),
        skew(signal),
        kurtosis(signal),
        np.ptp(signal),
        peak_freq,
        band_power,
        spectral_centroid,
    ]


def load_csi_csv(csi_file_path):
    data = pd.read_csv(csi_file_path)
    data["date_time"] = pd.to_datetime(data["date_time"])
    data = data.sort_values("date_time").reset_index(drop=True)
    return data


def split_respiration_windows(data, window_seconds=30, step_seconds=1):
    if data.empty:
        return []

    windows = []

    start = data["date_time"].min()
    end = data["date_time"].max()

    current = start

    while current + pd.Timedelta(seconds=window_seconds) <= end:
        next_time = current + pd.Timedelta(seconds=window_seconds)

        window = data[
            (data["date_time"] >= current) &
            (data["date_time"] < next_time)
        ].copy()

        if len(window) >= 50:
            windows.append(window)

        current = current + pd.Timedelta(seconds=step_seconds)

    return windows


def predict_respiration_rate_from_window(window_data):
    if len(window_data) < 50:
        return None

    data_sub = ProcessCSI(window_data)
    data_sub = remove_dc_offset(data_sub)

    signal = np.mean(data_sub, axis=1)

    time_range = (
        window_data["date_time"].iloc[-1]
        - window_data["date_time"].iloc[0]
    ).total_seconds()

    if time_range <= 0:
        return None

    fs = len(signal) / time_range

    signal = bandpass_filter(signal, fs, 0.1, 0.5)

    features = extract_features(signal, fs)
    features_scaled = scaler.transform([features])

    rr = model.predict(features_scaled)[0]

    return round(float(rr), 2)


def get_respiration_windows(csi_file_path):
    data = load_csi_csv(csi_file_path)
    return split_respiration_windows(
        data,
        window_seconds=WINDOW_SECONDS,
        step_seconds=STEP_SECONDS
    )

