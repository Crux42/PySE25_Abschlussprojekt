from pathlib import Path
import re

tile_dir = Path("data/image-tiles")
pattern = r"tile_([\w_]+)_z\d+_x\d+_y\d+_([\d]{8}_[\d]{4})\.png"

timestamps = set()

for tile in tile_dir.rglob("*.png"):
    match = re.match(pattern, tile.name)
    if match:
        timestamps.add(match.group(2))

print(f"🕒 Gefundene Timestamps: {sorted(timestamps)}")