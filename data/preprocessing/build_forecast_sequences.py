import pandas as pd
import numpy as np
from pathlib import Path

CSV_PATH = Path("training_data_all_cities.csv")
OUTPUT_PATH = Path("forecast_sequences.npz")

SEQUENCE_LENGTH = 1008  # 3 Wochen à 48 Schritte/Tag
FORECAST_STEPS = [24, 48, 72, 96, 120]  # = +12h, +24h, ..., +60h
TARGET_FEATURES = ["temperature", "pressure", "wind_speed", "clouds", "rain", "snow", "humidity"]
TARGET_CITY = "Linz"

print("📥 Lade CSV …")
df = pd.read_csv(CSV_PATH)

# Zeitstempel verarbeiten
df.insert(0, "timestamp", pd.to_datetime(df["rounded_time"], format="%Y%m%d_%H%M"))
df = df.sort_values("timestamp").reset_index(drop=True)
df.drop(columns=["rounded_time"], inplace=True)

# Zielspalten: z. B. Linz_temperature, Linz_pressure, …
target_cols = [f"{TARGET_CITY}_{f}" for f in TARGET_FEATURES]

# Eingabe-Features: alle außer timestamp
input_cols = [col for col in df.columns if col != "timestamp"]
X_all = df[input_cols].values  # [T, input_dim]
y_all = df[target_cols].values  # [T, 7]

# Sequenzen bauen
X_seq, y_seq = [], []
for i in range(SEQUENCE_LENGTH, len(df) - max(FORECAST_STEPS)):
    x = X_all[i - SEQUENCE_LENGTH:i]  # 3 Wochen Input
    y = [y_all[i + step] for step in FORECAST_STEPS]  # Ziel-Zeitpunkte
    if any(np.isnan(y_t).any() for y_t in y):
        continue
    X_seq.append(x)
    y_seq.append(y)

X_seq = np.array(X_seq)  # [N, 1008, input_dim]
y_seq = np.array(y_seq)  # [N, 5, 7]

print("✅ Sequenzen erstellt:", X_seq.shape, y_seq.shape)
np.savez_compressed(OUTPUT_PATH, X=X_seq, y=y_seq)
print(f"💾 Gespeichert unter {OUTPUT_PATH}")