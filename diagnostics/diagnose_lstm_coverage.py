import pandas as pd
import numpy as np
from pathlib import Path

CSV_PATH = "training_data_all_cities.csv"
SEQUENCE_LENGTH = 6
REQUIRED_LAYERS = ["clouds", "precipitation", "pressure", "temp", "wind"]
MERGED_IMG_DIR = Path("data/merged_tiles")

# 1. CSV laden
df = pd.read_csv(CSV_PATH)
assert "rounded_time" in df.columns, "Spalte 'rounded_time' fehlt"
ziel_features = [col for col in df.columns if col.startswith("Linz_")]
input_df = df.drop(columns=ziel_features)

# 2. Alle Zeitstempel extrahieren
timestamps = df["rounded_time"].tolist()
valid_timestamps = []
invalid_reasons = []

for i in range(len(df) - SEQUENCE_LENGTH):
    time_seq = timestamps[i:i+SEQUENCE_LENGTH]  # Eingabesequenz
    target_time = timestamps[i+SEQUENCE_LENGTH]  # Zielzeit

    # a) Prüfe, ob die Sequenz vollständig aufeinanderfolgend ist
    if len(set(time_seq)) < SEQUENCE_LENGTH:
        invalid_reasons.append((target_time, "unvollständige Sequenz"))
        continue

    # b) Prüfe, ob alle Layerbilder für das Zielzeitfenster existieren
    all_layers_found = True
    for layer in REQUIRED_LAYERS:
        image_path = MERGED_IMG_DIR / layer / f"{layer}_{target_time}.png"
        if not image_path.exists():
            all_layers_found = False
            invalid_reasons.append((target_time, f"{layer}-Bild fehlt"))
            break

    if all_layers_found:
        valid_timestamps.append(target_time)

# 3. Ergebnisse anzeigen
print(f"\n✅ Gültige Zielzeitpunkte: {len(valid_timestamps)}")
print(f"❌ Ungültige oder unvollständige Zeitpunkte: {len(invalid_reasons)}\n")

for ts, reason in invalid_reasons[:15]:
    print(f"{ts}: {reason}")

# Optional als CSV speichern
pd.DataFrame(invalid_reasons, columns=["timestamp", "reason"]).to_csv("invalid_lstm_timestamps.csv", index=False)