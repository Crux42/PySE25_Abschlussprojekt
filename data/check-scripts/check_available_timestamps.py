from pathlib import Path
from collections import defaultdict

# Lokaler Pfad zu deinem Projekt
base_path = Path("/Users/kraxi/Documents/Coding/WIFI/PySE25_Abschlussprojekt/data/merged-images")
layers = ["clouds", "precipitation", "pressure", "temp", "wind"]

layer_timestamps = {}
all_timestamps = defaultdict(set)

# 1. Timestamps pro Layer sammeln
for layer in layers:
    folder = base_path / layer
    if not folder.exists():
        print(f"⚠️ Ordner nicht gefunden: {folder}")
        continue
    timestamps = set()
    for file in folder.glob(f"{layer}_*.png"):
        try:
            timestamp_part = file.stem.split("_")[-1]
            timestamps.add(timestamp_part)
            all_timestamps[timestamp_part].add(layer)
        except IndexError:
            continue
    layer_timestamps[layer] = timestamps
    print(f"✅ {layer}: {len(timestamps)} Zeitstempel gefunden")

# 2. Schnittmenge: vollständige Timestamps
if len(layer_timestamps) == len(layers):
    common = set.intersection(*layer_timestamps.values())
    print(f"\n✅ {len(common)} vollständige Timestamps (alle 5 Layer):")
    for ts in sorted(common)[:10]:
        print(f"   - {ts}")

# 3. Fehlen aufschlüsseln
print("\n🔍 Unvollständige Timestamps:")
partial = {ts: layers_present for ts, layers_present in all_timestamps.items() if len(layers_present) < len(layers)}
for ts, present_layers in sorted(partial.items())[:10]:
    missing = [l for l in layers if l not in present_layers]
    print(f"   - {ts} fehlt: {', '.join(missing)}")
print(f"\n📉 Insgesamt {len(partial)} Zeitpunkte mit fehlenden Layern")
