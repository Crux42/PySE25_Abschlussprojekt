Schritt-für-Schritt-Anleitung für PythonAnywhere:

## 1. Richtige Konsole starten
- Gehe auf https://www.pythonanywhere.com/consoles
- Klicke "Start a new bash console"
Dort kannst du Shell-Befehle korrekt ausführen.

---

## 2. Virtuelle Umgebung einrichten (optional, empfohlen)
'''bash
python3.10 -m venv venv
source venv/bin/activate
'''

---

## 3. Benötigte Pakete installieren
'''bash
pip install requests python-dotenv google-cloud-storage
'''

---

## 4. Fehlerbehebung für 'ImportError: cannot import name 'storage''
Dieser Fehler bedeutet, dass entweder:
- das google-cloud-storage Paket nicht installiert ist oder
- es in einem falschen Pfad liegt (nicht in deiner Umgebung)
Mit Schritt 3 stellst du sicher, dass es in deiner Umgebung korrekt installiert ist.

---

## 5. Skript ausführen (Test)
'''bash
python WeatherToCloud.py
'''
Wenn das Skript korrekt durchläuft, kannst du zum Scheduler übergehen.

---

## 6. Scheduler-Task anlegen
1. Gehe zu "Tasks" in PythonAnywhere
2. Klicke auf „Add a new scheduled task“
3. Trage ein:
'''bash
/home/DEIN_USERNAME/venv/bin/python /home/DEIN_USERNAME/WeatherToCloud.py
'''
4. Wähle: Every 30 minutes