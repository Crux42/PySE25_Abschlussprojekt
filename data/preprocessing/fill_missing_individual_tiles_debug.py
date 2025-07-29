import os
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

# Konfiguration
INPUT_DIR = Path("data/image-tiles")
EXPECTED_X = range(6, 10)
EXPECTED_Y = range(4, 7)
EXPECTED_TILES = {(x, y) for x in EXPECTED_X for y in EXPECTED_Y}
TIME_INTERVAL = timedelta(minutes=30)

# Regex zum Parsen
TILE_PATTERN = re.compile(r"tile_(\w+)_z4_x(\d+)_y(\d+)_([\d]{8}_[\d]{6})\.png")

def parse_tile_filename(filename):
    match = TILE_PATTERN.match(filename)
    if not match:
        return None
    layer, x, y, timestamp = match.groups()
    return {
        "layer": layer,
        "x": int(x),
        "y": int(y),
        "timestamp": timestamp
    }

def round_to_30min(dt):
    total_minutes = dt.hour * 60 + dt.minute
    rounded_minutes = int((total_minutes + 15) / 30) * 30
    return dt.replace(hour=rounded_minutes // 60, minute=rounded_minutes % 60, second=0, microsecond=0)

def collect_tile_data(folder):
    tiles = defaultdict(lambda: defaultdict(dict))  # tiles[timestamp][(x,y)] = filepath
    for file in folder.glob("*.png"):
        parsed = parse_tile_filename(file.name)
        if not parsed:
            continue
        dt = datetime.strptime(parsed["timestamp"], "%Y%m%d_%H%M%S")
        rounded = round_to_30min(dt)
        key = (parsed["x"], parsed["y"])
        ts_str = rounded.strftime("%Y%m%d_%H%M")
        tiles[ts_str][key] = file
    return tiles

def fill_missing_individual_tiles():
    for category_folder in INPUT_DIR.iterdir():
        if not category_folder.is_dir():
            continue

        layer = category_folder.name.replace("-folder", "").lower()
        print(f"\n🔍 Prüfe Layer: {layer}")

        tile_data = collect_tile_data(category_folder)
        timestamps = sorted(datetime.strptime(ts, "%Y%m%d_%H%M") for ts in tile_data.keys())
        if not timestamps:
            print("⚠️ Keine Daten vorhanden.")
            continue

        first = timestamps[0]
        last = timestamps[-1]
        current = first

        while current <= last:
            ts_str = current.strftime("%Y%m%d_%H%M")
            current_tiles = tile_data.get(ts_str, {})
            found_keys = set(current_tiles.keys())
            missing_coords = EXPECTED_TILES - found_keys

            print(f"\n🕒 Timestamp: {ts_str}")
            print(f"✔ Gefundene Tiles: {len(found_keys)}/12")
            if missing_coords:
                print(f"❌ Fehlende Tiles: {sorted(missing_coords)}")
                prev_ts = current - TIME_INTERVAL
                prev_str = prev_ts.strftime("%Y%m%d_%H%M")
                prev_tiles = tile_data.get(prev_str, {})

                for coord in sorted(missing_coords):
                    if coord in prev_tiles:
                        x, y = coord
                        src = prev_tiles[coord]
                        new_name = f"tile_{layer}_z4_x{x}_y{y}_{ts_str}00.png"
                        dst = category_folder / new_name
                        shutil.copy(src, dst)
                        print(f"  ➕ Kopiert {coord} von {prev_str} → {ts_str}")
                        tile_data[ts_str][coord] = dst  # wichtig für Folgezeitpunkte
                    else:
                        print(f"  ⚠️ Kein Ersatz für {coord} bei {ts_str}")
            else:
                print("✅ Vollständig")

            current += TIME_INTERVAL

if __name__ == "__main__":
    fill_missing_individual_tiles()