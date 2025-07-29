# data/check-scripts/repair_pngs.py

from PIL import Image, UnidentifiedImageError
from pathlib import Path

def repair_pngs_in_folder(folder: Path, create_backup: bool = False):
    repaired = 0
    skipped = 0
    failed = 0

    for file in folder.rglob("*.png"):
        try:
            Image.open(file).verify()
            continue  # Datei ist gültig → nichts tun
        except Exception:
            print(f"🛠️ Versuche Reparatur: {file}")

        try:
            img = Image.open(file).convert("RGBA")
            img.save(file)
            print(f"✔️ Repariert: {file}")
            repaired += 1
        except Exception as e:
            print(f"❌ Fehlgeschlagen: {file} → {e}")
            failed += 1

    print("\n🧾 Zusammenfassung:")
    print(f"   Repariert:  {repaired}")
    print(f"   Fehlerhaft: {failed}")
    print(f"   Gesamt:     {repaired + failed}")

if __name__ == "__main__":
    root_folder = Path("data/image-tiles")
    repair_pngs_in_folder(root_folder, create_backup=False)