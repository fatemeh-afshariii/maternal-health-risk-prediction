"""
generate_data.py

This script creates a synthetic (fake, computer-generated) dataset that
simulates health measurements of pregnant women, along with a risk level
(low, mid, high). No real patient data is used -- everything here is
randomly generated for learning and portfolio purposes.

HOW TO RUN THIS FILE:
    python generate_data.py
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_PATIENTS = 500

age = np.random.randint(15, 45, N_PATIENTS)
systolic_bp = np.random.randint(90, 180, N_PATIENTS)          # Systolic Blood Pressure
diastolic_bp = np.random.randint(60, 110, N_PATIENTS)         # Diastolic Blood Pressure
blood_sugar = np.round(np.random.uniform(6.0, 19.0, N_PATIENTS), 1)   # Blood Sugar (mmol/L)
body_temp_f = np.round(np.random.uniform(98.0, 103.0, N_PATIENTS), 1) # Body Temperature (F)
heart_rate = np.random.randint(60, 110, N_PATIENTS)           # Heart Rate (bpm)

# Simple rule to decide risk level based on the numbers above
# (This mimics real clinical risk factors in a simplified way)
risk_score = (
    (systolic_bp > 140).astype(int) * 2
    + (diastolic_bp > 90).astype(int) * 2
    + (blood_sugar > 11).astype(int) * 2
    + (body_temp_f > 100.5).astype(int)
    + (heart_rate > 100).astype(int)
    + (age > 35).astype(int)
)

risk_level = np.select(
    [risk_score <= 1, risk_score <= 3],
    ["low risk", "mid risk"],
    default="high risk",
)

df = pd.DataFrame({
    "Age": age,
    "SystolicBP": systolic_bp,
    "DiastolicBP": diastolic_bp,
    "BloodSugar": blood_sugar,
    "BodyTemp_F": body_temp_f,
    "HeartRate": heart_rate,
    "RiskLevel": risk_level,
})

df.to_csv("maternal_health_data.csv", index=False)
print(f"Created synthetic dataset with {len(df)} rows.")
print(df.head())
print("\nRisk level counts:")
print(df["RiskLevel"].value_counts())
