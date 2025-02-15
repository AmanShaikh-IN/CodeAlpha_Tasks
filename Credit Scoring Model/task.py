import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from google.colab import drive
from IPython.display import display

drive.mount('/content/drive', force_remount=True)

dataset_path = "/content/drive/MyDrive/ProjectData/Credit_Score_Clean.csv"
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

raw_data = pd.read_csv(dataset_path)

print(raw_data.head(10), "\n")
print(raw_data.info(), "\n")
print(raw_data.describe(include='all'), "\n")

data = raw_data.copy()

# .cat.codes works better than labelencoding
data['Occupation'] = data['Occupation'].astype('category').cat.codes
data['Credit_Mix'] = data['Credit_Mix'].astype('category').cat.codes
data['Payment_of_Min_Amount'] = data['Payment_of_Min_Amount'].astype('category').cat.codes
data['Payment_Behaviour'] = data['Payment_Behaviour'].astype('category').cat.codes

# Mapping
credit_score_mapping = {'Poor': 0, 'Standard': 1, 'Good': 2}
data['Credit_Score'] = data['Credit_Score'].map(credit_score_mapping)

# Feature/Target Split
X = data.drop(columns=['Credit_Score'])
y = data['Credit_Score']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=365)

# Model 1
rf_model = RandomForestClassifier(random_state=365, n_jobs=-1, n_estimators=100)
rf_model.fit(X_train, y_train)

# Model 2
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=365)
xgb_model.fit(X_train, y_train)

# Testing
rf_pred = rf_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)

def evaluate_model(y_test, y_pred, model_name):
    print(f"\n--- {model_name} Performance ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.2f}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.2f}")
    
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Poor", "Standard", "Good"])
    disp.plot()
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()

# Evaluating
evaluate_model(y_test, rf_pred, "Random Forest")
evaluate_model(y_test, xgb_pred, "XGBoost")

def predict_credit_worthiness(model, *features):
    input_data = pd.DataFrame([features], columns=X.columns)
    credit_score_code = model.predict(input_data)[0]
    score_mapping = {0: "Poor", 1: "Standard", 2: "Good"}
    return score_mapping.get(credit_score_code, "Unknown")

# Sample Predictions
predicted_score_xgb = predict_credit_worthiness(xgb_model, 30, 1, 50000, 2, 3, 5, 2, 1, 0, 10, 2, 1, 10000, 0.3, 0, 200, 300, 2, 1000, 120)
predicted_score_rf = predict_credit_worthiness(rf_model, 30, 1, 50000, 2, 3, 5, 2, 1, 0, 10, 2, 1, 10000, 0.3, 0, 200, 300, 2, 1000, 120)

print(f"Predicted Credit Worthiness Score (XGBoost): {predicted_score_xgb}")
print(f"Predicted Credit Worthiness Score (Random Forest): {predicted_score_rf}")
