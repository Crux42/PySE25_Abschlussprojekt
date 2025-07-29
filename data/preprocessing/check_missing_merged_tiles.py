from pathlib import Path
from datetime import datetime, timedelta
import re

MERGED_DIR = Path("data/merged_tiles")
TIMESTAMP_RE = re.compile(r"^(\w+)_(\d{8}_\d{4})\.png$")

def parse_filename(filename):
    match = TIMESTAMP_RE.match(filename)
    if match:
        category, timestamp = match.groups()
        dt = datetime.strptime(timestamp, "%Y%m%d_%H%M")
        return category, dt
    return None, None

def check_missing_tiles():
    for category_dir in MERGED_DIR.iterdir():
        if not category_dir.is_dir():
            continue

        category = category_dir.name
        timestamps = []

        for file in category_dir.glob("*.png"):
            cat, dt = parse_filename(file.name)
            if cat == category and dt:
                timestamps.append(dt)

        if not timestamps:
            print(f"\nKeine Dateien in Kategorie '{category}'")
            continue

        timestamps = sorted(timestamps)
        first, last = timestamps[0], timestamps[-1]

        print(f"\nKategorie: {category}")
        print(f"Zeitraum: {first} → {last} ({len(timestamps)} Bilder)")

        expected = set()
        current = first
        while current <= last:
            expected.add(current)
            current += timedelta(minutes=30)

        existing = set(timestamps)
        missing = sorted(expected - existing)

        if missing:
            print(f"Fehlende Zeitpunkte ({len(missing)}):")
            for m in missing:
                print(f"  - {m.strftime('%Y-%m-%d %H:%M')}")
        else:
            print("Keine Lücken gefunden.")

if __name__ == "__main__":
    check_missing_tiles()