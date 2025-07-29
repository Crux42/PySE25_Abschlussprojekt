import logging
import os
import requests
import sqlite3
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
from google.cloud import storage


# ======================================================================
# API-Keys laden
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
# Logging initialisieren
bucket_name = "weathermaps2-bucket"
log_blob_path = "logfile.txt"
local_logfile = "local_logfile.txt"
bucket = storage_client.bucket(bucket_name)
log_blob = bucket.blob(log_blob_path)

if log_blob.exists():
    log_blob.download_to_filename(local_logfile)
else:
    open(local_logfile, "w").close()

logging.basicConfig(
    filename=local_logfile,
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger()
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', '%Y-%m-%d %H:%M:%S')
console.setFormatter(formatter)
logger.addHandler(console)

# ======================================================================
# Parameter definieren
layers = ["clouds_new", "precipitation_new", "pressure_new", "wind_new", "temp_new"]
zoom = 4
x_min, x_max = 6, 9
y_min, y_max = 4, 6
tile_size = 256
total_width = (x_max - x_min + 1) * tile_size
total_height = (y_max - y_min + 1) * tile_size
timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

# ======================================================================
# Tiles direkt nach Download hochladen

for layer in layers:
    try:
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                url = f"https://tile.openweathermap.org/map/{layer}/{zoom}/{x}/{y}.png?appid={API_KEY_OPENWEATHERMAP}"
                filename = f"tile_{layer}_z{zoom}_x{x}_y{y}_{timestamp}.png"
                local_path = Path(filename)

                response = requests.get(url)
                if response.status_code != 200:
                    logger.warning(f"[{layer}] Fehler beim Download: Statuscode {response.status_code} für Tile x={x}, y={y}")
                    continue

                with open(local_path, "wb") as f:
                    f.write(response.content)
                logger.info(f"[{layer}] Tile gespeichert: {filename}")

                folder = f"{layer.replace('_new', '')}-folder"
                destination_blob_name = f"{folder}/{filename}"
                blob = bucket.blob(destination_blob_name)
                blob.upload_from_filename(local_path)
                logger.info(f"Tile hochgeladen: gs://{bucket_name}/{destination_blob_name}")

                os.remove(local_path)

    except Exception as e:
        logger.error(f"[{layer}] Fehler beim Verarbeiten der Tiles: {e}")

        
# ======================================================================
# NUMERISCHE WETTERDATEN SPEICHERN

base_path = os.path.dirname(__file__)
numerical_db_path = os.path.join(base_path, "weather_data.sqlite3")
conn = sqlite3.connect(numerical_db_path)
cursor = conn.cursor()

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
        logger.info(f"Tabelle erweitert: {', '.join(missing_columns)}")

cursor.execute("CREATE INDEX IF NOT EXISTS idx_city_time ON weather_data (city, timestamp);")
conn.commit()
logger.info("Index gesetzt.")

# ======================================================================
# WETTERDATEN ABRUFEN

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

        timestamp_city = datetime.utcfromtimestamp(data["dt"]).isoformat()
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
        ''', (city, timestamp_city, temperature, pressure, humidity, wind_speed, wind_deg, clouds, rain, snow))

        logger.info(f"Wetterdaten gespeichert für {city}: {timestamp_city}")

    except Exception as e:
        logger.error(f"Fehler bei {city}: {e}")

conn.commit()
conn.close()

# ======================================================================
# Wetterdaten-DB nach GCS hochladen

destination_blob_name = "numerical-data/weather_data.sqlite3"
blob = storage_client.bucket(bucket_name).blob(destination_blob_name)
blob.upload_from_filename(numerical_db_path)
logger.info(f"Wetterdaten-Datenbank hochgeladen: gs://{bucket_name}/{destination_blob_name}")

# ======================================================================
# Logfile nach GCS hochladen

log_blob.upload_from_filename(local_logfile)
logger.info(f"Logfile aktualisiert: gs://{bucket_name}/{log_blob_path}")
