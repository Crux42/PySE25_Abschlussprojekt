import os
import re
import json
from PIL import Image
from datetime import datetime, timedelta
from collections import defaultdict

# Konfiguration
input_dir = "data/image-tiles"
output_dir = "data/merged_tiles"
cache_file = os.path.join(output_dir, "merge_cache.json")
log_file = os.path.join(output_dir, "missing_tile_groups.log")
debug_file = os.path.join(output_dir, "debug_tile_grouping.log")
tile_size = (256, 256)
tile_coords = [(x, y) for y in range(4, 7) for x in range(6, 10)]

filename_re = re.compile(r"tile_(\w+)_z4_x(\d+)_y(\d+)_(\d{8}_\d{6})\.png")

def parse_filename(filename):
    match = filename_re.match(filename)
    if match:
        category, x, y, timestamp = match.groups()
        dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
        return category, int(x), int(y), dt
    return None

def round_to_nearest_half_hour(dt):
    minute = dt.minute
    if minute < 15:
        return dt.replace(minute=0, second=0, microsecond=0)
    elif minute < 45:
        return dt.replace(minute=30, second=0, microsecond=0)
    else:
        dt += timedelta(hours=1)
        return dt.replace(minute=0, second=0, microsecond=0)

def timestamp_key(dt):
    return dt.strftime("%Y%m%d_%H%M")

def group_tiles_by_rounded_timestamp(files, debug_lines):
    groups = defaultdict(list)
    for file in files:
        parsed = parse_filename(file)
        if not parsed:
            debug_lines.append(f"Datei übersprungen (ungültiger Name): {file}")
            continue
        category, x, y, dt = parsed
        rounded_dt = round_to_nearest_half_hour(dt)
        groups[rounded_dt].append((file, x, y, dt))
        debug_lines.append(f"Zugeordnet: {file} -> {rounded_dt} ({x},{y})")
    return groups

def load_cache():
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2)

def log_missing_group(key, category, tiles_found):
    expected_coords = set(tile_coords)
    found_coords = set((x, y) for _, x, y, _ in tiles_found)
    missing_coords = sorted(expected_coords - found_coords)
    with open(log_file, "a") as f:
        f.write(f"{category}_{key}: {len(found_coords)}/12 Tiles vorhanden, fehlend: {missing_coords}\n")

def merge_tiles_for_group(group, category, group_time, cache):
    key = f"{category}_{timestamp_key(group_time)}"
    category_out_dir = os.path.join(output_dir, category)
    os.makedirs(category_out_dir, exist_ok=True)
    out_path = os.path.join(category_out_dir, f"{key}.png")

    if key in cache or os.path.exists(out_path):
        print(f"⏭ Bereits verarbeitet: {key}")
        return

    canvas = Image.new("RGBA", (tile_size[0] * 4, tile_size[1] * 3))
    tiles_placed = 0

    for file, x, y, _ in group:
        img_path = os.path.join(input_dir, f"{category}-folder", file)
        if not os.path.exists(img_path):
            print(f"Datei fehlt auf Platte: {img_path}")
            continue
        try:
            tile = Image.open(img_path)
            pos_x = (x - 6) * tile_size[0]
            pos_y = (y - 4) * tile_size[1]
            canvas.paste(tile, (pos_x, pos_y))
            tiles_placed += 1
        except Exception as e:
            print(f"Fehler beim Öffnen {file}: {e}")
            continue

    if tiles_placed == 12:
        canvas.save(out_path)
        cache[key] = True
        save_cache(cache)
        print(f"✔ Gespeichert: {out_path}")
    else:
        print(f"⚠ Unvollständig ({tiles_placed}/12) für {key}")
        log_missing_group(timestamp_key(group_time), category, group)

def main():
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(log_file):
        os.remove(log_file)
    if os.path.exists(debug_file):
        os.remove(debug_file)
    
    cache = load_cache()
    debug_lines = []

    for category_folder in os.listdir(input_dir):
        full_path = os.path.join(input_dir, category_folder)
        if not os.path.isdir(full_path):
            continue

        category = category_folder.replace("-folder", "")
        print(f"\n>>> Kategorie: {category}")

        all_files = os.listdir(full_path)
        tile_groups = group_tiles_by_rounded_timestamp(all_files, debug_lines)

        for group_time, group_tiles in tile_groups.items():
            merge_tiles_for_group(group_tiles, category, group_time, cache)

    with open(debug_file, "w") as f:
        f.write("\n".join(debug_lines))

if __name__ == "__main__":
    main()
