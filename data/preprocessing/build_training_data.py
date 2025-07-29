import sqlite3
import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np
import os

CITIES = ["Linz", "Prag", "Brünn", "Bratislava", "Maribor", "Villach",
          "Innsbruck", "München", "Regensburg", "Amsterdam", "La Rochelle",
          "Genua", "Zadar", "Belgrad", "Warschau"]

DB_PATH = "data/numerical-data/Weather_data.sqlite3"
TABLE_NAME = "weather_data_with_rounded"
MERGED_IMG_DIR = Path("data/merged_tiles")
LAYERS = ["clouds", "precipitation", "pressure", "temp", "wind"]

OUTPUT_CSV = "training_data_all_cities.csv"


def image_to_features(image_path):
    try:
        img = Image.open(image_path).resize((64, 64)).convert("L")
        return np.array(img).flatten()
    except Exception as e:
        print(f"Fehler beim Laden von {image_path}: {e}")
        return None


def get_numeric_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()

    df["city"] = df["city"].str.strip().str.normalize("NFC")
    df = df[df["city"].isin(CITIES)]
    df["rounded_time"] = df["rounded_time"].astype(str)
    return df


def build_training_rows(df):
    grouped = df.groupby("rounded_time")
    rows = []
    timestamps = []

    for ts_str, group in grouped:
        if len(group["city"].unique()) < len(CITIES):
            continue

        images = []
        all_found = True
        for layer in LAYERS:
            img_path = MERGED_IMG_DIR / layer / f"{layer}_{ts_str}.png"
            if not img_path.exists():
                all_found = False
                break
            features = image_to_features(img_path)
            if features is None:
                all_found = False
                break
            images.append(features)

        if not all_found:
            continue

        group = group.set_index("city")
        features = []

        for city in CITIES:
            row = group.loc[city]
            features.extend([
                row["temperature"], row["pressure"], row["humidity"], row["wind_speed"],
                row["wind_deg"], row["clouds"], row["rain"], row["snow"]
            ])

        image_features = np.concatenate(images)
        full_row = features + image_features.tolist()
        rows.append(full_row)
        timestamps.append(ts_str)

    return rows, timestamps


def main():
    print("📦 Lade numerische Wetterdaten …")
    df_numeric = get_numeric_data()
    print("🔎 Städte im Datensatz:", sorted(df_numeric["city"].unique()))

    print("Erzeuge Trainingszeilen …")
    rows, timestamps = build_training_rows(df_numeric)

    print(f"Trainingsdaten gespeichert: {OUTPUT_CSV}")
    print(f"Anzahl Zeilen: {len(rows)}")

    columns = []
    for city in CITIES:
        columns += [f"{city}_{feature}" for feature in [
            "temperature", "pressure", "humidity", "wind_speed", "wind_deg", "clouds", "rain", "snow"
        ]]
    for layer in LAYERS:
        columns += [f"{layer}_px_{i}" for i in range(64 * 64)]

    df_out = pd.DataFrame(rows, columns=columns)
    df_out["rounded_time"] = timestamps
    df_out.to_csv(OUTPUT_CSV, index=False)

if __name__ == "__main__":
    main()