# Maternal Health Risk Prediction with Machine Learning

## Overview
This project uses a synthetic (computer-generated, non-real) dataset that simulates
common clinical measurements collected during pregnancy -- age, blood pressure,
blood sugar, body temperature, and heart rate -- to predict a maternal health
**risk level** (low, mid, or high) using a machine learning model.

The project was built as a hands-on learning exercise combining a clinical/midwifery
background with practical data science and machine learning skills.

⚠️ **Note:** All data in this project is synthetically generated for
educational and portfolio purposes. No real patient data is used.

## Project Structure
```
├── generate_data.py     # Creates the synthetic dataset
├── analysis.py           # Exploratory charts + machine learning model
├── requirements.txt      # Required Python libraries
└── README.md
```

## Methodology
1. A synthetic dataset of 500 simulated patients is generated with six
   clinical features: Age, Systolic BP, Diastolic BP, Blood Sugar, Body
   Temperature, and Heart Rate.
2. A rule-based risk score assigns each patient a Risk Level
   (low / mid / high risk), simulating simplified clinical decision logic.
3. A **Random Forest Classifier** is trained to predict Risk Level from the
   clinical features.
4. Model performance is evaluated with a classification report and confusion
   matrix, and feature importance is visualized to show which factors matter
   most for the prediction.

## How to Run
```bash
pip install -r requirements.txt
python generate_data.py
python analysis.py
```

## Outputs
- `maternal_health_data.csv` — the generated synthetic dataset
- `risk_level_counts.png` — distribution of risk levels
- `age_vs_bp.png` — relationship between age, blood pressure, and risk
- `feature_importance.png` — which clinical factors matter most
- `confusion_matrix.png` — model performance

## Future Directions
- Apply the same pipeline to real, de-identified clinical data (with proper
  ethical approval)
- Try additional models (Logistic Regression, XGBoost) and compare performance
- Build an interactive web app (e.g., with Streamlit) for real-time risk
  prediction

## Author
Fatemeh Afshari - A midwifery graduate with a research interest in AI applications in
women's health.
