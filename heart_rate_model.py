import ast
import joblib
import numpy as np
import pandas as pd

from scipy.signal import butter, filtfilt


MODEL_PATH = "heart_rate_model.pkl"

WINDOW_SECONDS = 60
TOP_K = 5

LOW_BPM = 50
HIGH_BPM = 100

model = joblib.load(MODEL_PATH)
print("Heart rate model loaded successfully")


def parse_csi_data(data_string):
    try:
        return np.array(ast.literal_eval(data_string), dtype=float)
    except Exception:
        return None


def calculate_amplitude_from_iq(iq_array):
    if iq_array is None or len(iq_array) < 2:
        return None

    if len(iq_array) % 2 != 0:
        iq_array = iq_array[:-1]

    i_vals = iq_array[0::2]
    q_vals = iq_array[1::2]

    return np.sqrt(i_vals ** 2 + q_vals ** 2)


def load_csi_csv(path):
    df = pd.read_csv(path)
    df["date_time"] = pd.to_datetime(df["date_time"])

    times = []
    amplitudes = []

    for _, row in df.iterrows():
        raw = parse_csi_data(row["data"])
        amp = calculate_amplitude_from_iq(raw)

        if amp is None:
            continue

        times.append(row["date_time"])
        amplitudes.append(amp)

    return pd.DataFrame({
        "date_time": times,
        "amplitude": amplitudes
    })


def split_windows(df, window_seconds=60, step_seconds=1):
    if df.empty:
        return []

    df = df.sort_values("date_time").reset_index(drop=True)

    windows = []
    start = df["date_time"].min()
    end = df["date_time"].max()

    current = start

    while current + pd.Timedelta(seconds=window_seconds) <= end:
        next_time = current + pd.Timedelta(seconds=window_seconds)

        window = df[
            (df["date_time"] >= current) &
            (df["date_time"] < next_time)
        ].copy()

        if not window.empty:
            windows.append(window)

        # overlap step
        current = current + pd.Timedelta(seconds=step_seconds)

    return windows


def hampel_filter(signal, window_size=5, n_sigmas=3):
    signal = np.asarray(signal, dtype=float)
    filtered = signal.copy()

    for i in range(len(signal)):
        start = max(0, i - window_size)
        end = min(len(signal), i + window_size + 1)

        window = signal[start:end]
        median = np.median(window)
        mad = np.median(np.abs(window - median))

        if mad == 0:
            continue

        threshold = n_sigmas * 1.4826 * mad

        if abs(signal[i] - median) > threshold:
            filtered[i] = median

    return filtered


def bandpass_filter(signal, fs):
    lowcut = LOW_BPM / 60.0
    highcut = HIGH_BPM / 60.0

    nyquist = 0.5 * fs

    if lowcut <= 0 or highcut >= nyquist or lowcut >= highcut:
        return signal

    b, a = butter(
        4,
        [lowcut / nyquist, highcut / nyquist],
        btype="band"
    )

    return filtfilt(b, a, signal)


def normalize_signal(signal):
    signal = np.asarray(signal, dtype=float)
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-6)


def estimate_fft_hr(signal, fs):
    signal = np.asarray(signal, dtype=float)

    if len(signal) < 16:
        return 0.0

    fft_size = 2 ** int(np.ceil(np.log2(len(signal))))

    spectrum = np.fft.rfft(signal, n=fft_size)
    freqs = np.fft.rfftfreq(fft_size, d=1 / fs)
    power = np.abs(spectrum) ** 2

    low_hz = LOW_BPM / 60.0
    high_hz = HIGH_BPM / 60.0

    mask = (freqs >= low_hz) & (freqs <= high_hz)

    if not np.any(mask):
        return 0.0

    peak_freq = freqs[mask][np.argmax(power[mask])]

    return peak_freq * 60.0


def extract_features(signal, fs):
    fft_hr = estimate_fft_hr(signal, fs)

    return [
        np.mean(signal),
        np.std(signal),
        np.var(signal),
        np.min(signal),
        np.max(signal),
        np.median(signal),
        np.percentile(signal, 25),
        np.percentile(signal, 75),
        fft_hr
    ]


def build_features(processed_matrix, fs):
    feature_vector = []
    fft_hrs = []

    for i in range(processed_matrix.shape[1]):
        sig = processed_matrix[:, i]

        feature_vector.extend(extract_features(sig, fs))
        fft_hrs.append(estimate_fft_hr(sig, fs))

    fft_hrs = np.array(fft_hrs, dtype=float)

    feature_vector.extend([
        np.mean(fft_hrs),
        np.std(fft_hrs),
        np.median(fft_hrs),
        np.min(fft_hrs),
        np.max(fft_hrs)
    ])

    return np.array(feature_vector, dtype=float)


def preprocess_window(window):
    try:
        amplitude_matrix = np.vstack(window["amplitude"].values)
    except Exception:
        return None, None

    if amplitude_matrix.shape[0] < 30:
        return None, None

    duration = (
        window["date_time"].max() -
        window["date_time"].min()
    ).total_seconds()

    if duration <= 0:
        return None, None

    fs = len(window) / duration

    variances = np.var(amplitude_matrix, axis=0)
    selected_indices = np.argsort(variances)[-TOP_K:]
    selected_matrix = amplitude_matrix[:, selected_indices]

    selected_matrix = selected_matrix - np.mean(selected_matrix, axis=0)

    processed = []

    for i in range(selected_matrix.shape[1]):
        sig = selected_matrix[:, i]

        sig = hampel_filter(sig)
        sig = bandpass_filter(sig, fs)
        sig = normalize_signal(sig)

        processed.append(sig)

    processed_matrix = np.array(processed).T

    return processed_matrix, fs


def predict_heart_rate_from_window(window):
    processed_matrix, fs = preprocess_window(window)

    if processed_matrix is None:
        return None

    feature_vector = build_features(processed_matrix, fs)

    if len(feature_vector) != 50:
        print("Invalid feature count. Expected 50.")
        return None

    prediction = model.predict([feature_vector])[0]

    return round(float(prediction), 2)


def get_heart_rate_windows(csi_file_path):
    csi_df = load_csi_csv(csi_file_path)
    windows = split_windows(csi_df, window_seconds=60, step_seconds=1)
    return windows