import sqlite3
import pandas as pd
from datetime import datetime, timedelta

DB_PATH = "data/numerical-data/Weather_data.sqlite3"
TABLE_NAME = "weather_data"
ROUNDED_COL = "rounded_time"

def round_to_half_hour(dt):
    discard = timedelta(minutes=dt.minute % 30, seconds=dt.second, microseconds=dt.microsecond)
    dt -= discard
    if discard >= timedelta(minutes=15):
        dt += timedelta(minutes=30)
    return dt.replace(second=0, microsecond=0)

def main():
    print("📥 Lade Daten aus SQLite …")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)

    print("Runde Zeitstempel …")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df[ROUNDED_COL] = df["timestamp"].apply(round_to_half_hour)
    df[ROUNDED_COL] = df[ROUNDED_COL].dt.strftime("%Y%m%d_%H%M")

    print("Speichere Tabelle als temporäre Kopie …")
    df.to_sql(f"{TABLE_NAME}_with_rounded", conn, if_exists="replace", index=False)

    print(f"Gespeichert als Tabelle '{TABLE_NAME}_with_rounded'")
    conn.close()

if __name__ == "__main__":
    main()
