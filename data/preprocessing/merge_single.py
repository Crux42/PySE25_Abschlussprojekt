
import os
import re
import sys
import argparse
from PIL import Image
from datetime import datetime
from collections import defaultdict

# Konfiguration
input_dir = "data/image-tiles"
output_dir = "data/merged_tiles"
tile_size = (256, 256)
tile_coords = [(x, y) for y in range(4, 7) for x in range(6, 10)]

# Regex zum Parsen
filename_re = re.compile(r"tile_(\w+)_z4_x(\d+)_y(\d+)_([\d]{8}_[\d]{6})\.png")

def parse_filename(filename):
    match = filename_re.match(filename)
    if match:
        category, x, y, timestamp = match.groups()
        dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
        return category, int(x), int(y), dt
    return None

def merge_single(category, timestamp_str):
    print(f"🔧 Merge: Kategorie = {category}, Zeit = {timestamp_str}")
    ts_dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M")
    matched_files = []

    folder = os.path.join(input_dir, f"{category}-folder")
    if not os.path.exists(folder):
        print(f"Ordner nicht gefunden: {folder}")
        return

    for file in os.listdir(folder):
        parsed = parse_filename(file)
        if parsed:
            _, x, y, dt = parsed
            rounded = dt.replace(second=0, microsecond=0)
            rounded = rounded.replace(minute=(0 if dt.minute < 15 or dt.minute < 45 and dt.minute < 30 else 30))
            key = dt.strftime("%Y%m%d_%H%M")
            if key == timestamp_str:
                matched_files.append((file, x, y))

    if len(matched_files) != 12:
        print(f"Nur {len(matched_files)}/12 Tiles gefunden → Abbruch.")
        return

    canvas = Image.new("RGBA", (tile_size[0] * 4, tile_size[1] * 3))
    for file, x, y in matched_files:
        img_path = os.path.join(folder, file)
        tile = Image.open(img_path)
        pos_x = (x - 6) * tile_size[0]
        pos_y = (y - 4) * tile_size[1]
        canvas.paste(tile, (pos_x, pos_y))

    # Speichern
    out_dir = os.path.join(output_dir, category)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{category}_{timestamp_str}.png")
    canvas.save(out_path)
    print(f"Gespeichert: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True, help="Kategorie, z. B. temp, clouds")
    parser.add_argument("--timestamp", required=True, help="Timestamp im Format YYYYMMDD_HHMM")
    args = parser.parse_args()
    merge_single(args.category, args.timestamp)
