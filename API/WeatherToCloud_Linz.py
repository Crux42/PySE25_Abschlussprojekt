import logging
import os
import math
import requests
import sqlite3
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
from google.cloud import storage

# ======================================================================

# API-Keys

env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

API_KEY_OPENWEATHERMAP = os.getenv("API_KEY_OPENWEATHERMAP")
if not API_KEY_OPENWEATHERMAP:
    raise RuntimeError("OpenWeatherMap API-Key fehlt!")

google_key_path = Path(__file__).parent / "ServiceKey_GoogleCloud.json"
if not google_key_path.exists():
    raise RuntimeError("Google Cloud Key fehlt!")

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(google_key_path)
storage_client = storage.Client()

# ======================================================================

# LOGGING

bucket_name = "weathermaps-bucket"
log_blob_path = "logfile.txt"
local_logfile = "local_logfile.txt"
bucket = storage_client.bucket(bucket_name)
log_blob = bucket.blob(log_blob_path)

# Logfile anlegen oder laden
if log_blob.exists():
    log_blob.download_to_filename(local_logfile)
else:
    open(local_logfile, "w").close()

# Logging konfigurieren
logging.basicConfig(
    filename=local_logfile,
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger()

# Logging-Konsole
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', '%Y-%m-%d %H:%M:%S')
console.setFormatter(formatter)
logger.addHandler(console)

# ======================================================================

# PARAMETER FÜR KARTEN

layers = ["clouds_new", "precipitation_new", "pressure_new", "wind_new", "temp_new"]
zoom = 4
lat = 48.30639
lon = 14.28611

def latlon_to_tile(lat, lon, zoom):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x_tile = int((lon + 180.0) / 360.0 * n)
    y_tile = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x_tile, y_tile

x, y = latlon_to_tile(lat, lon, zoom)

# ======================================================================

# DOWNLOAD DER KARTEN

for layer in layers:
    try:
        url = f"https://tile.openweathermap.org/map/{layer}/{zoom}/{x}/{y}.png?appid={API_KEY_OPENWEATHERMAP}"
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"karte_{layer}_z{zoom}_x{x}_y{y}_{timestamp}.png"
        local_path = Path(filename)

        response = requests.get(url)
        if response.status_code != 200:
            logger.warning(f"[{layer}] Fehler beim Download: Statuscode {response.status_code}")
            continue

        with open(local_path, "wb") as f:
            f.write(response.content)
        logger.info(f"[{layer}] Karte gespeichert: {filename}")

        # Upload zu GCS in Layer-Ordner
        destination_blob_name = f"{layer.replace('_new', '')}-folder/{filename}"
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(local_path)
        logger.info(f"[{layer}] Erfolgreich in GCS hochgeladen: gs://{bucket_name}/{destination_blob_name}")

        # Lokale Datei optional löschen
        os.remove(local_path)

    except Exception as e:
        logger.error(f"[{layer}] Fehler beim Verarbeiten: {e}")

# ======================================================================

# DB FÜR NUMERISCHE WETTERDATEN

# Datenbank vorbereiten
base_path = os.path.dirname(__file__)
numerical_db_path = os.path.join(base_path, "weather_data.sqlite3")
conn = sqlite3.connect(numerical_db_path)
cursor = conn.cursor()

# Tabelle prüfen und ggf. erweitern
cursor.execute("PRAGMA table_info(weather_data)")
columns = [col[1] for col in cursor.fetchall()]


if not columns:
    cursor.execute('''
        CREATE TABLE weather_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            city TEXT,
            timestamp TEXT,
            temperature REAL,
            pressure INTEGER,
            humidity INTEGER,
            wind_speed REAL,
            wind_deg INTEGER,
            clouds INTEGER,
            rain REAL,
            snow REAL
        )
    ''')
    conn.commit()
    logger.info("Tabelle 'weather_data' neu erstellt.")
else:
    expected_columns = {"city", "timestamp", "temperature", "pressure", "humidity", "wind_speed", "wind_deg", "clouds", "rain", "snow"}
    missing_columns = expected_columns - set(columns)
    for col in missing_columns:
        cursor.execute(f"ALTER TABLE weather_data ADD COLUMN {col} REAL")
    if missing_columns:
        conn.commit()
        logger.info(f"Tabelle 'weather_data' um Spalten erweitert: {', '.join(missing_columns)}")

# Index hinzufügen für bessere Abfragegeschwindigkeit
cursor.execute("CREATE INDEX IF NOT EXISTS idx_city_time ON weather_data (city, timestamp);")
conn.commit()
logger.info("Index 'idx_city_time' wurde gesetzt.")

# ======================================================================

# WETTERDATEN ABRUFEN UND SPEICHERN

cities = [
    "Linz", "Prag", "Brünn", "Bratislava", "Maribor", "Villach",
    "Innsbruck", "München", "Regensburg", "Amsterdam",
    "La Rochelle", "Genua", "Zadar", "Belgrad", "Warschau"
]


for city in cities:
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY_OPENWEATHERMAP}&units=metric"
        response = requests.get(url)
        data = response.json()

        timestamp = datetime.utcfromtimestamp(data["dt"]).isoformat()
        temperature = data["main"]["temp"]
        pressure = data["main"]["pressure"]
        humidity = data["main"]["humidity"]
        wind_speed = data["wind"]["speed"]
        wind_deg = data["wind"].get("deg")
        clouds = data["clouds"]["all"]
        rain = data.get("rain", {}).get("1h", 0.0)
        snow = data.get("snow", {}).get("1h", 0.0)

        cursor.execute('''
            INSERT INTO weather_data (city, timestamp, temperature, pressure, humidity, wind_speed, wind_deg, clouds, rain, snow)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (city, timestamp, temperature, pressure, humidity, wind_speed, wind_deg, clouds, rain, snow))

        logger.info(f"Wetterdaten gespeichert für {city}: {timestamp}")

    except Exception as e:
        logger.error(f"Fehler bei {city}: {e}")

conn.commit()
conn.close()


# Upload der Wetterdaten
bucket_name = "weathermaps-bucket"
destination_blob_name = "numerical-data/weather_data.sqlite3"
blob = storage_client.bucket(bucket_name).blob(destination_blob_name)
blob.upload_from_filename(numerical_db_path)
logger.info(f"Wetterdaten-Datenbank hochgeladen: gs://{bucket_name}/{destination_blob_name}")


# ======================================================================

# LOGFILE AKTUALISIEREN

log_blob.upload_from_filename(local_logfile)
logger.info(f"Logfile aktualisiert in GCS: gs://{bucket_name}/{log_blob_path}")