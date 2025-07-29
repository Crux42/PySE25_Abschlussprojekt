import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from model.standard.train_model import MLP
from sklearn.preprocessing import StandardScaler

CSV_PATH = "training_data_all_cities.csv"
MODEL_PATH = "weather_model.pth"

# 1. Daten laden
df = pd.read_csv(CSV_PATH)
ziel_features = [col for col in df.columns if col.startswith("Linz_")]
X = df.drop(columns=ziel_features)
y = df[ziel_features]

# 2. Modell + Skalierer laden
checkpoint = torch.load(MODEL_PATH, weights_only=False)
scaler_X = checkpoint['scaler_X']
scaler_y = checkpoint['scaler_y']
model = MLP(input_dim=X.shape[1], output_dim=len(ziel_features))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 3. Daten skalieren
X_scaled = scaler_X.transform(X)
y_scaled = scaler_y.transform(y)

# 4. Vorhersage
tensor_X = torch.tensor(X_scaled, dtype=torch.float32)
with torch.no_grad():
    pred_scaled = model(tensor_X).numpy()

# 5. Rückskalieren
pred = scaler_y.inverse_transform(pred_scaled)
y_true = y.values

# 6. Metriken berechnen
mae = mean_absolute_error(y_true, pred)
mse = mean_squared_error(y_true, pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, pred)

print("\n📊 Modell-Evaluation:")
print(f"MAE  (Mean Absolute Error): {mae:.2f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
print(f"R²   (Erklärte Varianz): {r2:.3f}")