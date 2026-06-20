import joblib
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


# ==============================
# LOAD POSTURE MODELS
# ==============================
knn_model = joblib.load("posture models/smartpulse_knn_model.pkl")
pca_model = joblib.load("posture models/smartpulse_pca.pkl")
scaler_model = joblib.load("posture models/smartpulse_scaler.pkl")

print("Posture model loaded successfully")


# ==============================
# SETTINGS
# ==============================
WINDOW_SECONDS = 30
STEP_SECONDS = 2


# ==============================
# LOAD CSI CSV
# ==============================
def load_csi_csv(csi_file_path):
    data = pd.read_csv(csi_file_path)

    if "Date_Time" in data.columns:
        data["Date_Time"] = pd.to_datetime(data["Date_Time"], errors="coerce")
    elif "date_time" in data.columns:
        data["Date_Time"] = pd.to_datetime(data["date_time"], errors="coerce")
    else:
        raise ValueError("Date_Time column not found")

    data = data.dropna(subset=["Date_Time"])
    data = data.sort_values("Date_Time").reset_index(drop=True)

    return data


# ==============================
# RAW CSI TO AMPLITUDE
# ==============================
def get_amp(df):
    df = df.copy()

    if "data" in df.columns:
        csi_col = "data"
    elif "CSI_DATA" in df.columns:
        csi_col = "CSI_DATA"
    else:
        raise ValueError("CSI data column not found. Need 'data' or 'CSI_DATA'")

    amp_list = []
    valid_times = []

    for _, row in df.iterrows():
        try:
            parts = (
                str(row[csi_col])
                .replace("[", "")
                .replace("]", "")
                .replace('"', "")
                .split(",")
            )

            parts = [p.strip() for p in parts if p.strip() != ""]
            parts = np.array(parts, dtype=np.float64)

            if len(parts) < 128:
                continue

            parts = parts[:128]

            im_csi = parts[::2]
            re_csi = parts[1::2]

            amp = np.sqrt(im_csi**2 + re_csi**2)

            # Keep same 52 subcarriers used in training
            amp_52 = np.concatenate((amp[6:32], amp[33:59]))

            amp_list.append(amp_52)
            valid_times.append(row["Date_Time"])

        except Exception:
            continue

    if len(amp_list) == 0:
        return None

    amp_array = np.array(amp_list)

    # DC removal per file/session
    amp_dc_removed = amp_array - np.mean(amp_array, axis=0)

    amp_df = pd.DataFrame(
        amp_dc_removed,
        columns=[f"sc_{i}" for i in range(amp_dc_removed.shape[1])]
    )

    amp_df["Date_Time"] = valid_times

    return amp_df


# ==============================
# TIMESTAMP-BASED FEATURE EXTRACTION
# ==============================
def extract_features_with_time_window(
    amp_df,
    window_seconds=WINDOW_SECONDS,
    step_seconds=STEP_SECONDS
):
    amp_df = amp_df.copy()
    amp_df = amp_df.sort_values("Date_Time").reset_index(drop=True)

    subcarrier_cols = [col for col in amp_df.columns if col.startswith("sc_")]

    start_time = amp_df["Date_Time"].min()
    last_time = amp_df["Date_Time"].max()

    features = []
    current_start = start_time

    while current_start + pd.Timedelta(seconds=window_seconds) <= last_time:
        current_end = current_start + pd.Timedelta(seconds=window_seconds)

        window_df = amp_df[
            (amp_df["Date_Time"] >= current_start) &
            (amp_df["Date_Time"] < current_end)
        ]

        if len(window_df) < 10:
            current_start += pd.Timedelta(seconds=step_seconds)
            continue

        window_features = {}

        for col in subcarrier_cols:
            values = window_df[col].values

            window_features[f"{col}_mean"] = np.mean(values)
            window_features[f"{col}_max"] = np.max(values)
            window_features[f"{col}_min"] = np.min(values)
            window_features[f"{col}_var"] = np.var(values)
            window_features[f"{col}_skew"] = skew(values)
            window_features[f"{col}_range"] = np.max(values) - np.min(values)
            window_features[f"{col}_median"] = np.median(values)
            window_features[f"{col}_kurtosis"] = kurtosis(values)

        features.append(window_features)

        current_start += pd.Timedelta(seconds=step_seconds)

    return pd.DataFrame(features)


# ==============================
# PREDICT POSTURE FROM FILE
# ==============================
def predict_posture_from_file(csi_file_path):
    data = load_csi_csv(csi_file_path)

    amp_df = get_amp(data)

    if amp_df is None or amp_df.empty:
        print("No valid CSI amplitude data")
        return None

    features = extract_features_with_time_window(
        amp_df,
        window_seconds=WINDOW_SECONDS,
        step_seconds=STEP_SECONDS
    )

    features = features.replace([np.inf, -np.inf], np.nan).dropna()

    if features.empty:
        print("No valid posture feature windows")
        return None

    scaled_features = scaler_model.transform(features)
    pca_features = pca_model.transform(scaled_features)

    predictions = knn_model.predict(pca_features)

    pred_counts = pd.Series(predictions).value_counts()

    print("Posture window counts:")
    print(pred_counts)

    final_posture = pred_counts.idxmax()

    return str(final_posture)