# Groundwater Depletion Detection & Prediction using Machine Learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------------
# 1. Load Dataset
# -------------------------
# Your dataset should include:
# Year | Groundwater_Level | Rainfall | Temperature | Pumping_Rate | Soil_Moisture

data = pd.read_csv("groundwater_data.csv")

print("Dataset Preview:")
print(data.head())

# -------------------------
# 2. Preprocessing
# -------------------------
data = data.dropna()

X = data[['Rainfall', 'Temperature', 'Pumping_Rate', 'Soil_Moisture']]
y = data['Groundwater_Level']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------
# 3. Machine Learning Model
# -------------------------
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# -------------------------
# 4. Evaluate Model
# -------------------------
print("\nModel Accuracy (R² Score):", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# -------------------------
# 5. Detect Depletion Trend
# -------------------------
data['Predicted'] = model.predict(X)

plt.plot(data['Year'], data['Groundwater_Level'], label="Actual Levels")
plt.plot(data['Year'], data['Predicted'], label="Predicted Trend", linestyle='dashed')

plt.xlabel("Year")
plt.ylabel("Groundwater Level (m)")
plt.title("Groundwater Depletion Trend")
plt.legend()
plt.show()

# -------------------------
# 6. Identify High-Risk Depletion Zones
# -------------------------
data['Drop'] = data['Groundwater_Level'].diff()

THRESHOLD = -0.5   # Customize based on region

high_risk = data[data['Drop'] < THRESHOLD]
print("\nHigh-Risk Depletion Periods:")
print(high_risk[['Year', 'Groundwater_Level', 'Drop']])