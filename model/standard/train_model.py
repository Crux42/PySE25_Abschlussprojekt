import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Pfad zur CSV
CSV_PATH = "training_data_all_cities.csv"
MODEL_PATH = "weather_model.pth"

# Ziel-Features für Linz
ZIEL_FEATURES = [
    "temperature", "pressure", "humidity", "wind_speed",
    "wind_deg", "clouds", "rain", "snow"
]

# 1. Daten laden
df = pd.read_csv(CSV_PATH)

# 2. Eingabe- und Zielwerte definieren
X = df.drop(columns=[f"Linz_{z}" for z in ZIEL_FEATURES])
y = df[[f"Linz_{z}" for z in ZIEL_FEATURES]]

# 3. Standardisieren
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# 4. Train/test split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# 5. In Torch-Tensoren umwandeln
tensor_X_train = torch.tensor(X_train, dtype=torch.float32)
tensor_y_train = torch.tensor(y_train, dtype=torch.float32)
tensor_X_val = torch.tensor(X_val, dtype=torch.float32)
tensor_y_val = torch.tensor(y_val, dtype=torch.float32)

train_ds = TensorDataset(tensor_X_train, tensor_y_train)
val_ds = TensorDataset(tensor_X_val, tensor_y_val)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=32)

# 6. Modell definieren
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.net(x)

model = MLP(input_dim=X.shape[1], output_dim=len(ZIEL_FEATURES))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# 7. Training
EPOCHS = 30
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
        val_pred = model(tensor_X_val)
        val_loss = loss_fn(val_pred, tensor_y_val)
    print(f"Epoch {epoch+1}/{EPOCHS} → Validierungsverlust: {val_loss.item():.4f}")

# 8. Modell + Skalierer speichern
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler_X': scaler_X,
    'scaler_y': scaler_y,
    'ziel_features': ZIEL_FEATURES
}, MODEL_PATH)

print(f"\n✅ Modell gespeichert unter {MODEL_PATH}")
