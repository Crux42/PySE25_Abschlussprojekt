import sqlite3
import pandas as pd

conn = sqlite3.connect("data/numerical-data/Weather_data.sqlite3")
df = pd.read_sql_query("SELECT * FROM weather_data", conn)
df["timestamp"] = pd.to_datetime(df["timestamp"])

city_counts = df.groupby("timestamp")["city"].count()
valid_timestamps = city_counts[city_counts == 15]
print("✅ Gültige Timestamps mit 15 Städten:", len(valid_timestamps))
print(valid_timestamps.head())


from datetime import timedelta
from pathlib import Path

MERGED_IMG_ROOT = Path("data/merged-images")
LAYERS = ["clouds", "precipitation", "pressure", "temp", "wind"]

def has_all_images(ts):
    ts_str = ts.strftime("%Y%m%d%H%M%S")
    for layer in LAYERS:
        img_path = MERGED_IMG_ROOT / layer / f"merged_{layer}_{ts_str}.png"
        if not img_path.exists():
            return False
    return True

count = 0
for ts in valid_timestamps.index:
    rounded = ts.round("30min")
    if has_all_images(rounded):
        count += 1

print("📸 Zeitpunkte mit 15 Städten und allen 5 Layer-Bildern:", count)
