# PySE25_Abschlussprojekt

# 1. Projektübersicht

**Projektname:**  
*CLIMATE CRYSTAL ORB*

**Kurzbeschreibung:**  
*Mittels historischer Wetterdaten und KI-Auswertung soll das zukünftige Wetter für Linz vorhergesagt werden. Dabei werden die Wetterdaten aus den folgenden Städten aufgezeichnet: Linz, Prag, Brünn, Bratislava, Maribor, Villach, Innsbruck, München, Regensburg, Amsterdam, La Rochelle, Genua, Zadar, Belgrad, Warschau. Zusätzlich werden Wetterkarten von Europa über einen bestimmten Zeitraum aufgezeichnet. Das Modell (Letztstand: LSTM decoder-encoder attention) errechnet aus den letzten 3 Wochen, bei 30-Minuten-Intervallen, das Wetter der nächsten 60 Stunden für Linz.*

**Zielgruppe / Use Case:**  
*Das Projekt dient lediglich als persönlichen Test.*

---

# 2. Projektziele & Akzeptanzkriterien

| Ziel | Akzeptanzkriterium |
|------|--------------------|
| Speicherung Kartenmaterial | Periodisches Downloaden von Kartenmaterial mit Wetterdaten und Upload in Cloud |
| Speicherung num. Wetterdaten | Periodisches Downloaden numerischer Wetterdaten mehrerer europäischer Städte als SQLite3-Datei und Upload in Cloud |
| Erstellung Trainingsdatensatz | Automatische Erzeugung eines einheitlichen CSV aus numerischen Daten und Speicherung im Projektordner |
| Zusammenfügen von Karten | Alle 12 Tiles pro Zeitstempel korrekt mergen zu einem Gesamtbild |
| KI-Vorhersagemodell | Vorhersage von Temperatur, Druck, Wind, Regen, Schnee, Wolken und Feuchte für Linz mittels LSTM- oder Encoder-Decoder-Modell |
| Live-Forecast | Modell erzeugt aktuelle Wetterprognose für Linz auf Basis der letzten 3 Wochen Daten |

---

# 3. User Stories & Aufgabenliste

| #  | User Story                                                                 | Task-Beschreibung                                            | Optim. (h) | Realist. (h) | Pessi. (h) | Mittlere Schätzung (h) | Priorität (H/M/L) |
|----|----------------------------------------------------------------------------|---------------------------------------------------------------|------------|--------------|------------|------------------------|-------------------|
| 1  | Als Entwickler möchte ich alle Wetterkarten tiles automatisch mergen, um visuell nutzbare Gesamtbilder zu erhalten. | Tile-Merger-Skript schreiben, Timestamp-Prüfung, Fehlerbehandlung | 3          | 4            | 6          | 4.2                    | Hoch              |
| 2  | Als Entwickler möchte ich numerische Wetterdaten mehrerer Städte laden, um ein Trainingsdataset zu erstellen. | Datenbankanbindung, Export als CSV, Feature-Auswahl          | 2          | 3            | 4          | 3.0                    | Hoch              |
| 3  | Als Entwickler möchte ich ein LSTM- oder Encoder-Decoder-Modell trainieren, um Wetter für Linz vorherzusagen. | Modellstruktur, Training, Validierung, Persistierung         | 5          | 8            | 12         | 8.2                    | Hoch              |
| 4  | Als Nutzer möchte ich eine einfache Streamlit-UI, um eine Live-Prognose anzuzeigen. | GUI mit Ladebalken, Modellintegration, Anzeige               | 2          | 3            | 5          | 3.5                    | Mittel            |
| 5  | Als Entwickler möchte ich ein Skript zur Vorhersage der kommenden 48h starten, um die Prognosewerte für Linz zu erzeugen. | Forecast-Skript, Dateninput laden, Modell laden, Ausgabe CSV | 2          | 3            | 4          | 3.0                    | Hoch              |
| 6  | Als Nutzer möchte ich eine grafische Benutzeroberfläche, um den Code auszuführen und die Ergebnisse anzuzeigen. | GUI mit Streamlit erstellen: Buttons, Eingaben, Ausgabe-Anzeige | 2          | 3            | 5          | 3.5                    | Mittel            |




> **Hinweis zur PERT-Schätzung:**  
> Mittlere = (Opt.+4·Real.+Pess.)/6

---

> **Hinweis zur Schätzung nach PERT:**  
> Mittlere = (Opt.+4·Real.+Pess.)/6

---

# 4. Zeit-Tracking

Während der Umsetzung führt ihr ein Log, in dem ihr die tatsächlich investierte Zeit erfasst.

| Datum       | Task #  | Geplante Zeit (h) | Tatsächlich (h) | Abweichung (h) | Kommentar                                                   |
|-------------|---------|-------------------|------------------|----------------|-------------------------------------------------------------|
| 2025-04-25  | 1       | 4,0               | 4,0              | 0              | Recherche                                                   |
| 2025-05-16  | 2       | 4,0               | 4,0              | 0              | Recherche                                                   |
| 2025-05-17  | 3       | 4,0               | 4,0              | 0              | Beginn API                                                  |
| 2025-05-23  | 4       | 2,0               | 2,0              | 0              | API                                                         |
| 2025-06-17  | 5       | 4,0               | 5,0              | +1             | data/preprocessing                                          |
| 2025-06-25  | 6       | 5,0               | 5,0              | 0              | DL begonnen - Probleme mit build_training_data.py (kein Output) |
| 2025-06-29  | 7       | 6,0               | 14,0             | +8             | Modellarchitektur und Debugging                             |
| 2025-06-30  | 8       | 4,0               | 10,0             | +6             | Training & Evaluation                                       |
| 2025-07-01  | 9       | 4,0               | 10,0             | +6             | Forecast-Skript, Modell laden, Tests                        |
| 2025-07-03  | 10       | 4,0               | 9,0              | +5             | Streamlit-Integration                                       |
| 2025-07-04  | 11       | 0,0               | 5,0              | +5             | Aufräumen, Validierung, Code-Kommentare                    |


---

# 5. Technischer Stack & Tools

- **Programmiersprache(n):** Python 3.x
- **Frameworks / Bibliotheken:** PyTorch, Pandas, Matplotlib, SQLite3, Streamlit, PIL, logging, requests, dotenv, pathlib, datetime, google.cloud, numpy, sklearn, subprocess, time, threading
- **Datenbank / Persistenz:** SQLite3, Google Cloud Storage
- **Sonstiges:** Git, VSCode, cron, OpenWeatherMap API, Google Cloud API

---

# 6. Projektplan & Meilensteine

| Meilenstein             | Datum        | Abgeschlossen (✔️/❌) |
|-------------------------|--------------|-----------------------|
| Initiales Setup         | 2025-04-25   | ✔️                      |
| Basis-Features fertig   | 2025-06-28   | ✔️                      |
| Test- und Feedbackphase | 2025-07-01   | ✔️                      |
| Finaler Feinschliff     | 2025-07-04   | ✔️                      |
| Abgabe                  | 2025-07-05   | ✔️                      |

---

# 7. Risiken & Gegenmaßnahmen

| Risiko                                      | Wahrscheinlichkeit (H/M/L) | Auswirkung (H/M/L) | Gegenmaßnahme                                |
|---------------------------------------------|-----------------------------|--------------------|----------------------------------------------|
| Wetterkarten fehlen / Tiles nicht vollständig | M                           | H                  | Validierungsskript für Vollständigkeit einsetzen |
| Modell ungenau bei starker Wetteränderung   | H                           | M                  | Modell mit mehr historischen Daten versorgen |
| PNG-Dateien korrupt oder fehlerhaft         | L                           | M                  | Fallback-Mechanismus mit Logging             |



---

# 8. Abschluss-Reflexion

- **Reine Entwicklungszeit (Summe tatsächl. Stunden):** 80,0 h  
- **Geplante Zeit (Summe):** 40,0 h  
- **Durchschnittliche Abweichung je Task:** ca. +3,5 h  
- **Was lief gut?** Automatisierung der Datenerfassung, saubere Aufbereitung der numerischen Wetterdaten, Modularität des Codes  
- **Wo hättet ihr besser schätzen können?** Implementierung des Deep-Learning-Modells, GUI-Integration und Fehlerbehandlung  
- **Lessons Learned / Tipps für’s nächste Projekt:** Frühzeitig Validierungsskripte einbauen, zeitliche Puffer für spätere Phasen einplanen, Debugging-Aufwand nicht unterschätzen


---

# 9. (Optional) Bonus-/Erweiterungsideen

- Deployment eines Vorhersage-Dashboards  
- Integration Satellitenbilder (z. B. Sentinel API)  
- Wetter-API-Anbindung für Vergleich Live-Daten vs. Prognose

---