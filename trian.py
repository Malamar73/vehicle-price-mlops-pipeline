
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load vehicle sales data
df = pd.read_csv("data/vehicle_sales.csv")

# Separate features and target
X = df[["year", "mileage", "condition"]]
y = df["price"]

# Convert condition into numerical features
X = pd.get_dummies(X, columns=["condition"])

# Train the model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# Create models folder
Path("models").mkdir(exist_ok=True)

# Save trained model
joblib.dump(
    {
        "model": model,
        "features": X.columns.tolist()
    },
    "models/vehicle_price_model.joblib"
)

print("Model trained and saved successfully!")
