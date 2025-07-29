# predict_single.py

import torch
import pandas as pd
from model.dataset import WeatherDataset
from model.model_architecture import WeatherFusionModel
from torchvision import transforms
from PIL import Image
from datetime import datetime
from pathlib import Path

def load_latest_sample(csv_path: str, timestamp: str = None):
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if timestamp:
        target_time = pd.to_datetime(timestamp)
        df = df[df["timestamp"] == target_time]
        if df.empty:
            raise ValueError(f"Kein Eintrag gefunden für {timestamp}")
    else:
        df = df.sort_values("timestamp", ascending=False).head(1)

    return df.iloc[0]  # eine Zeile

def prepare_sample(row, image_columns, numeric_columns, transform):
    image_tensors = []
    for col in image_columns:
        img = Image.open(row[col]).convert("RGB")
        img_tensor = transform(img)
        image_tensors.append(img_tensor)

    image_tensor = torch.cat(image_tensors, dim=0).unsqueeze(0)  # [1, C, H, W]
    numeric_tensor = torch.tensor([row[col] for col in numeric_columns], dtype=torch.float32).unsqueeze(0)  # [1, F]
    return image_tensor, numeric_tensor

def predict(csv_path: str, model_path: str, timestamp: str = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Vorhersage mit Modell auf {device}")

    # Dummy-Dataset zum Zugriff auf Spalten + Transformationen
    dummy = WeatherDataset(csv_path)
    image_columns = dummy.image_columns
    numeric_columns = dummy.numeric_columns
    transform = dummy.transform

    row = load_latest_sample(csv_path, timestamp)
    image_tensor, numeric_tensor = prepare_sample(row, image_columns, numeric_columns, transform)

    model = WeatherFusionModel(
        num_numeric_features=len(numeric_columns),
        image_input_channels=len(image_columns) * 3
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        image_tensor, numeric_tensor = image_tensor.to(device), numeric_tensor.to(device)
        pred = model(image_tensor, numeric_tensor).item()

    ts = row["timestamp"]
    print(f"\nVorhersage für {ts}: {pred:.2f} °C")

if __name__ == "__main__":
    # timestamp z. B. "2025-06-17 12:00:00" – sonst nimmt er den aktuellsten Eintrag
    predict("training_data_all_cities.csv", "weather_model.pth")
