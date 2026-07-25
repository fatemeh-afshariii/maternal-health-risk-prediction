"""
analysis.py

This script:
1) Loads the synthetic maternal health dataset
2) Draws some charts to explore the data
3) Trains a simple Machine Learning model to predict Risk Level
4) Saves the charts as image files so you can look at them and put them
   in your GitHub project

HOW TO RUN THIS FILE (after generate_data.py):
    python analysis.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ---------- 1) Load the data ----------
df = pd.read_csv("maternal_health_data.csv")

# ---------- 2) Simple charts ----------
plt.figure(figsize=(6, 5))
df["RiskLevel"].value_counts().plot(kind="bar", color="#4C72B0")
plt.title("Number of Patients per Risk Level")
plt.xlabel("Risk Level")
plt.ylabel("Number of Patients")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("risk_level_counts.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 5))
for level, color in zip(["low risk", "mid risk", "high risk"], ["green", "orange", "red"]):
    subset = df[df["RiskLevel"] == level]
    plt.scatter(subset["Age"], subset["SystolicBP"], label=level, alpha=0.6, color=color)
plt.title("Age vs Systolic Blood Pressure by Risk Level")
plt.xlabel("Age")
plt.ylabel("Systolic Blood Pressure")
plt.legend()
plt.tight_layout()
plt.savefig("age_vs_bp.png", dpi=150)
plt.close()

# ---------- 3) Machine learning model ----------
features = ["Age", "SystolicBP", "DiastolicBP", "BloodSugar", "BodyTemp_F", "HeartRate"]
X = df[features]

le = LabelEncoder()
y = le.fit_transform(df["RiskLevel"])  # turns text labels into numbers

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Model performance report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# Which features matter most for the prediction?
importances = pd.Series(model.feature_importances_, index=features).sort_values()
plt.figure(figsize=(7, 5))
importances.plot(kind="barh", color="#55A868")
plt.title("Which Factors Matter Most for Predicting Risk?")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()

print("\nAll charts saved: risk_level_counts.png, age_vs_bp.png, "
      "confusion_matrix.png, feature_importance.png")
