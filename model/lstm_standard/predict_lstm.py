import torch
import numpy as np
from model.lstm_standard.train_lstm_model import LSTMModel

SEQUENCE_FILE = "lstm_sequences.npz"
MODEL_PATH = "lstm_weather_model.pth"

# 1. Sequenzen laden
npz = np.load(SEQUENCE_FILE)
X_seq = torch.tensor(npz["X"], dtype=torch.float32)
y_true = npz["y"]
timestamps = npz["timestamps"].astype(str)

# 2. Modell vorbereiten
timestep, input_size = X_seq.shape[1], X_seq.shape[2]
output_size = y_true.shape[1]

model = LSTMModel(input_size, hidden_size=128, output_size=output_size)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# 3. Benutzer-Zeitstempel eingeben
print(f"\n🕒 Verfügbare Zeitstempel (Beispiel): {timestamps[:5]}")
ts_input = input("Gib einen Zeitstempel ein (YYYYMMDD_HHMM) oder Enter für letzten: ")

if ts_input in timestamps:
    idx = np.where(timestamps == ts_input)[0][0]
else:
    print("⏳ Kein gültiger Zeitstempel eingegeben, nehme letzte Sequenz.")
    idx = len(X_seq) - 1

input_seq = X_seq[idx].unsqueeze(0)  # [1, T, F]
true_vals = y_true[idx]

# 4. Vorhersage
with torch.no_grad():
    pred = model(input_seq).squeeze(0).numpy()

# 5. Ergebnis anzeigen
print(f"\n🔮 LSTM-Vorhersage für Linz bei {timestamps[idx]}:")
ziel_labels = ["temperature", "pressure", "humidity", "wind_speed", "wind_deg", "clouds", "rain", "snow"]
for name, p, t in zip(ziel_labels, pred, true_vals):
    print(f"{name.capitalize():<12}: {p:.2f} ➔ Tatsächlich: {t:.2f}")