import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib

# Parameter
CSV_PATH = "training_data_all_cities.csv"
MODEL_PATH = "models/forecast_model.pt"
SCALER_PATH = "models/scaler_y.pkl"
SEQUENCE_LENGTH = 1008
OUTPUT_CSV = "live_forecast_result.csv"

target_names = ["Temp", "Druck", "Wind", "Wolken", "Regen", "Schnee", "Feuchte"]

# Modellarchitektur (wie im Training)
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

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        outputs, (hidden, _) = self.lstm(x)
        return outputs, hidden[-1]

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, steps):
        super().__init__()
        self.steps = steps
        self.attn = Attention(hidden_dim, hidden_dim)
        self.lstm_cell = nn.LSTMCell(output_dim + hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, encoder_outputs, hidden):
        batch_size = encoder_outputs.size(0)
        dec_input = torch.zeros(batch_size, self.fc.out_features).to(encoder_outputs.device)
        cell = torch.zeros_like(hidden)
        outputs = []

        for _ in range(self.steps):
            context = self.attn(encoder_outputs, hidden)
            lstm_input = torch.cat((dec_input, context), dim=1)
            hidden, cell = self.lstm_cell(lstm_input, (hidden, cell))
            dec_input = self.fc(hidden)
            outputs.append(dec_input.unsqueeze(1))

        return torch.cat(outputs, dim=1)

class ForecastModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, out_steps):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(output_dim, hidden_dim, out_steps)

    def forward(self, x):
        enc_out, enc_hidden = self.encoder(x)
        return self.decoder(enc_out, enc_hidden)

# Daten laden
df = pd.read_csv(CSV_PATH)
df["timestamp"] = pd.to_datetime(df["rounded_time"], format="%Y%m%d_%H%M")
df = df.sort_values("timestamp")
df.drop(columns=["rounded_time"], inplace=True)

input_cols = [col for col in df.columns if col != "timestamp"]
X_all = df[input_cols].values

# Letzte 3 Wochen extrahieren
X_live = X_all[-SEQUENCE_LENGTH:]  # [1008, input_dim]
X_live_tensor = torch.tensor(X_live[np.newaxis, :, :], dtype=torch.float32)  # [1, 1008, input_dim]

# Scaler laden
scaler_y = joblib.load(SCALER_PATH)

# Modell laden
input_dim = X_live_tensor.shape[2]
model = ForecastModel(input_dim=input_dim, output_dim=7, hidden_dim=256, out_steps=5)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# Vorhersage durchführen
with torch.no_grad():
    pred_scaled = model(X_live_tensor).numpy()
    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 7)).reshape(1, 5, 7)

# Ausgabe als Tabelle
forecast_df = pd.DataFrame(pred[0], columns=target_names)
forecast_df.insert(0, "Forecast_Hour", ["+12h", "+24h", "+36h", "+48h", "+60h"])
forecast_df.to_csv(OUTPUT_CSV, index=False)
print(f"Vorhersage gespeichert in {OUTPUT_CSV}")