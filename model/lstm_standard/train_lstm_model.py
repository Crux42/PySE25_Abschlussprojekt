import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Konfiguration
SEQUENCE_FILE = "lstm_sequences.npz"
MODEL_PATH = "lstm_weather_model.pth"
EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-4

# Zielwertnamen und Einheiten
TARGET_NAMES = ["Temperatur", "Druck", "Wind", "Wolken", "Regen", "Schnee", "Feuchte"]
TARGET_UNITS = ["°C", "hPa", "m/s", "%", "mm/h", "mm/h", "%"]

# 1. Daten laden
npz = np.load(SEQUENCE_FILE)
X = torch.tensor(npz["X"], dtype=torch.float32)  # [B, T, F]
y_raw = npz["y"]  # numpy array [B, Output]

# Nur erste 7 Spalten verwenden (letzte Spalte = Target_7 entfernen, falls vorhanden)
y_raw = y_raw[:, :7]

# Ziel-Skalierung
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y_raw)
y = torch.tensor(y_scaled, dtype=torch.float32)

# 2. Train/test split
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

# 3. LSTM-Modell
timestep, input_size = X.shape[1], X.shape[2]
output_size = y.shape[1]

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=3, dropout=0.5, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

model = LSTMModel(input_size, hidden_size=512, output_size=output_size)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# 4. Training mit Val Loss Tracking
for epoch in range(EPOCHS):
    model.train()
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val).numpy()
        val_true = y_val.numpy()
        val_loss = mean_squared_error(val_true, val_pred)
    print(f"Epoch {epoch+1}/{EPOCHS} → Val Loss (scaled): {val_loss:.4f}")

# 5. RMSE je Zielwert in Originaleinheiten anzeigen
with torch.no_grad():
    pred_scaled = model(X_val).numpy()
    y_val_scaled = y_val.numpy()
    pred_unscaled = scaler_y.inverse_transform(pred_scaled)
    y_val_unscaled = scaler_y.inverse_transform(y_val_scaled)

    rmse_all = np.sqrt(np.mean((pred_unscaled - y_val_unscaled) ** 2, axis=0))
    print("\n📊 RMSE je Zielwert:")
    for i, rmse in enumerate(rmse_all):
        name = TARGET_NAMES[i] if i < len(TARGET_NAMES) else f"Target_{i}"
        unit = TARGET_UNITS[i] if i < len(TARGET_UNITS) else ""
        print(f"{name:10}: {rmse:.2f} {unit}")

# 6. Modell speichern
torch.save(model.state_dict(), MODEL_PATH)
print(f"\n✅ LSTM-Modell gespeichert unter {MODEL_PATH}")