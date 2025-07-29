import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib
import os

# Konfiguration
DATA_PATH = "forecast_sequences.npz"
SCALER_PATH = "models/scaler_y.pkl"
EPOCHS = 30
BATCH_SIZE = 4
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Laden
npz = np.load(DATA_PATH)
X = torch.tensor(npz["X"], dtype=torch.float32)
y = torch.tensor(npz["y"], dtype=torch.float32)

# Ziel-Skalierung
target_names = ["Temp", "Druck", "Wind", "Wolken", "Regen", "Schnee", "Feuchte"]
scaler_y = StandardScaler()
sh = y.shape  # [N, 5, 7]
y_scaled = scaler_y.fit_transform(y.view(-1, sh[-1]).numpy()).reshape(sh)
y = torch.tensor(y_scaled, dtype=torch.float32)

# Scaler speichern
os.makedirs("models", exist_ok=True)
joblib.dump(scaler_y, SCALER_PATH)
print(f"📦 Scaler gespeichert unter {SCALER_PATH}")

# Split
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

# Attention-Modul
class Attention(nn.Module):
    def __init__(self, enc_dim, dec_dim):
        super().__init__()
        self.attn = nn.Linear(enc_dim + dec_dim, 1)

    def forward(self, encoder_outputs, hidden):
        seq_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = self.attn(torch.cat((encoder_outputs, hidden), dim=2)).squeeze(-1)
        weights = torch.softmax(energy, dim=1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context

# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        outputs, (hidden, _) = self.lstm(x)
        return outputs, hidden[-1]

# Decoder
class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, steps):
        super().__init__()
        self.steps = steps
        self.attn = Attention(hidden_dim, hidden_dim)
        self.lstm_cell = nn.LSTMCell(output_dim + hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, encoder_outputs, hidden):
        batch_size = encoder_outputs.size(0)
        dec_input = torch.zeros(batch_size, self.fc.out_features).to(DEVICE)
        cell = torch.zeros_like(hidden)
        outputs = []

        for _ in range(self.steps):
            context = self.attn(encoder_outputs, hidden)
            lstm_input = torch.cat((dec_input, context), dim=1)
            hidden, cell = self.lstm_cell(lstm_input, (hidden, cell))
            dec_input = self.fc(hidden)
            outputs.append(dec_input.unsqueeze(1))

        return torch.cat(outputs, dim=1)

# Modell-Wrapper
class ForecastModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, out_steps):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(output_dim, hidden_dim, out_steps)

    def forward(self, x):
        enc_out, enc_hidden = self.encoder(x)
        dec_out = self.decoder(enc_out, enc_hidden)
        return dec_out

input_dim = X.shape[2]
output_dim = y.shape[2]
out_steps = y.shape[1]
model = ForecastModel(input_dim, output_dim, hidden_dim=256, out_steps=out_steps).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# Training
for epoch in range(EPOCHS):
    model.train()
    for xb, yb in train_dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val.to(DEVICE))
        val_loss = loss_fn(val_pred, y_val.to(DEVICE))
    print(f"Epoch {epoch+1}/{EPOCHS} → Val Loss (scaled): {val_loss.item():.4f}")

# RMSE-Auswertung
with torch.no_grad():
    y_val_pred = model(X_val.to(DEVICE)).cpu().numpy()
    y_val_true = y_val.cpu().numpy()

    y_val_pred_orig = scaler_y.inverse_transform(y_val_pred.reshape(-1, 7)).reshape(y_val_pred.shape)
    y_val_true_orig = scaler_y.inverse_transform(y_val_true.reshape(-1, 7)).reshape(y_val_true.shape)

    print("\n📊 RMSE je Zeitschritt & Zielwert:")
    for t in range(out_steps):
        rmse_t = np.sqrt(np.mean((y_val_pred_orig[:, t, :] - y_val_true_orig[:, t, :])**2, axis=0))
        label = f"+{(t+1)*12}h"
        for i, rmse in enumerate(rmse_t):
            print(f"{label:5} – {target_names[i]:8}: {rmse:.2f}")

# Modell speichern
MODEL_PATH = "models/forecast_model.pt"
torch.save(model.state_dict(), MODEL_PATH)
print(f"\n✅ Modell gespeichert unter {MODEL_PATH}")