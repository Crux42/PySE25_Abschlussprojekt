from google.cloud import storage
from pathlib import Path
import os

# INKREMENETALER DOWNLOAD

def download_from_gcs_if_missing(bucket_name: str, gcs_folder: str, local_folder: str, file_ext: str = ""):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=gcs_folder)

    downloaded = 0
    skipped = 0

    for blob in blobs:
        if blob.name.endswith("/") or (file_ext and not blob.name.endswith(file_ext)):
            continue  # Skip Ordner oder irrelevante Dateien

        relative_path = Path(blob.name).relative_to(gcs_folder)
        local_path = Path(local_folder) / relative_path
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if local_path.exists():
            print(f"⏩ Datei bereits vorhanden, überspringe: {local_path}")
            skipped += 1
            continue

        blob.download_to_filename(local_path)
        print(f"⬇️  Heruntergeladen: {blob.name} → {local_path}")
        downloaded += 1

    print(f"\n✅ Download abgeschlossen: {downloaded} neu, {skipped} übersprungen.")

if __name__ == "__main__":
    # Auth-File setzen
    key_path = Path(__file__).parent.parent / "ServiceKey_GoogleCloud.json"
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(key_path.resolve())

    # SQLite-Datenbank holen
    download_from_gcs_if_missing(
        bucket_name="weathermaps2-bucket",
        gcs_folder="numerical-data",
        local_folder="data/numerical-data",
        file_ext=".sqlite3"
    )

    # Alle Karten-Ordner holen
    for layer in ["clouds-folder", "precipitation-folder", "pressure-folder", "temp-folder", "wind-folder"]:
        download_from_gcs_if_missing(
            bucket_name="weathermaps2-bucket",
            gcs_folder=layer,
            local_folder=f"data/image-tiles/{layer}",
            file_ext=".png"
        )