import pandas as pd
import numpy as np

# Konfiguration
CSV_PATH = "training_data_all_cities.csv"
SEQUENCE_LENGTH = 6  # z. B. 6 Schritte = 3h bei 30min Abstand
OUTPUT_PATH = "lstm_sequences.npz"

# 1. CSV laden
df = pd.read_csv(CSV_PATH)
ziel_features = [col for col in df.columns if col.startswith("Linz_")]

# 2. Zeitinformationen aus rounded_time extrahieren
if "rounded_time" not in df.columns:
    raise ValueError("Spalte 'rounded_time' fehlt in der CSV")

timestamps = pd.to_datetime(df["rounded_time"], format="%Y%m%d_%H%M")
df["hour"] = timestamps.dt.hour
df["minute"] = timestamps.dt.minute

# Zeit als zyklische Features (z. B. 0 Uhr = 0°, 12 Uhr = 180°)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)

# 3. Eingabe definieren (ohne Zielwerte, aber MIT Zeitfeatures)
input_columns = [col for col in df.columns if col not in ziel_features and col != "rounded_time"]
X = df[input_columns].values.astype(np.float32)
y = df[ziel_features].values.astype(np.float32)
timestamps_str = df["rounded_time"].astype(str).values

# 4. Sequenzen erstellen
X_seq = []
y_seq = []
ts_seq = []
for i in range(len(df) - SEQUENCE_LENGTH):
    X_seq.append(X[i:i+SEQUENCE_LENGTH])
    y_seq.append(y[i+SEQUENCE_LENGTH])
    ts_seq.append(timestamps_str[i+SEQUENCE_LENGTH])

X_seq = np.stack(X_seq)
y_seq = np.stack(y_seq)
ts_seq = np.array(ts_seq)

# 5. Speichern
np.savez_compressed(OUTPUT_PATH, X=X_seq, y=y_seq, timestamps=ts_seq)
print(f"✅ Sequenzen gespeichert: {OUTPUT_PATH}")
print(f"📊 Eingabeform: {X_seq.shape}, Ziel: {y_seq.shape}, Zeit: {ts_seq.shape}")
