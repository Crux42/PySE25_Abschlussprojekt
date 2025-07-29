import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from model.standard.train_model import MLP

CSV_PATH = "training_data_all_cities.csv"
MODEL_PATH = "weather_model.pth"

# 1. CSV laden
df = pd.read_csv(CSV_PATH)

# 2. Benutzer wählt Timestamp
print("\nVerfügbare Zeitstempel (Beispiel):")
print(df.head(5)["rounded_time"] if "rounded_time" in df.columns else df.index[:5])

chosen_ts = input("Gib einen Zeitstempel im Format 'YYYYMMTT_HHMM' ein (oder Enter für letzten): ")

if "rounded_time" in df.columns and chosen_ts:
    row_match = df[df["rounded_time"] == chosen_ts]
    if row_match.empty:
        print("Kein passender Zeitstempel gefunden. Verwende letzte Zeile.")
        data_row = df.iloc[[-1]]
    else:
        data_row = row_match.iloc[[0]]
else:
    data_row = df.iloc[[-1]]

# 3. Ziel-Features extrahieren
ziel_features = [col for col in data_row.columns if col.startswith("Linz_")]
X = data_row.drop(columns=ziel_features)

# 4. Modell + Skalierer laden
import torch.serialization
torch.serialization.add_safe_globals([StandardScaler])
checkpoint = torch.load(MODEL_PATH, weights_only=False)

scaler_X = checkpoint['scaler_X']
scaler_y = checkpoint['scaler_y']
model = MLP(input_dim=X.shape[1], output_dim=len(ziel_features))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 5. Skalieren & Vorhersagen
X_scaled = scaler_X.transform(X)
x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
with torch.no_grad():
    pred_scaled = model(x_tensor).numpy()

# 6. Rückskalieren
pred = scaler_y.inverse_transform(pred_scaled)
true_values = data_row[ziel_features].values[0]

# 7. Ergebnis ausgeben
print("\nVorhersage für Linz:")
for name, value, true_val in zip(ziel_features, pred[0], true_values):
    label = name.replace("Linz_", "")
    print(f"{label.capitalize():<12}: {value:.2f} ➔ Tatsächlich: {true_val:.2f}")