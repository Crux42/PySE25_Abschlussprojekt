import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from train_lstm_model import LSTMModel

SEQUENCE_FILE = "lstm_sequences.npz"
MODEL_PATH = "lstm_weather_model.pth"

# 1. Lade Sequenzen
npz = np.load(SEQUENCE_FILE)
X = torch.tensor(npz["X"], dtype=torch.float32)
y_true = npz["y"]

# 2. Modell initialisieren und laden
timestep, input_size = X.shape[1], X.shape[2]
output_size = y_true.shape[1]
model = LSTMModel(input_size, hidden_size=128, output_size=output_size)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# 3. Vorhersagen berechnen
with torch.no_grad():
    pred = model(X).numpy()

# 4. Evaluation
mae = mean_absolute_error(y_true, pred)
rmse = mean_squared_error(y_true, pred, squared=False)
r2 = r2_score(y_true, pred)

print("\n📊 LSTM-Modell-Evaluation:")
print(f"MAE  (Mean Absolute Error): {mae:.2f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
print(f"R²   (Erklärte Varianz): {r2:.3f}")