# Credit Scoring Model

## Overview
This project implements a credit scoring model using two machine learning classifiers: **Random Forest** and **XGBoost**. The model predicts a person's creditworthiness based on financial and behavioral attributes.

## Dataset
- The dataset is stored in `Data/Credit_Score_Clean.csv`.
- It contains financial data such as income, credit mix, loan information, and payment behavior.
- The target variable is `Credit_Score`, categorized into:
  - **Poor (0)**
  - **Standard (1)**
  - **Good (2)**

## Dependencies
Make sure you have the required libraries installed before running the script:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn xgboost
```

## Features Used
The model uses the following features:
- **Occupation** (Categorical, encoded as numerical values)
- **Annual Income**
- **Number of Bank Accounts**
- **Number of Credit Cards**
- **Interest Rate**
- **Number of Loans**
- **Delay from Due Date**
- **Number of Delayed Payments**
- **Changed Credit Limit**
- **Number of Credit Inquiries**
- **Credit Mix** (Categorical, encoded)
- **Outstanding Debt**
- **Credit Utilization Ratio**
- **Payment of Minimum Amount** (Categorical, encoded)
- **Total EMI Per Month**
- **Amount Invested Monthly**
- **Payment Behavior** (Categorical, encoded)
- **Monthly Balance**
- **Credit History Age (Months)**

## Model Training
Two classifiers are trained:
1. **Random Forest Classifier**
2. **XGBoost Classifier**

```python
rf_model = RandomForestClassifier(random_state=365, n_jobs=-1, n_estimators=100)
rf_model.fit(X_train, y_train)

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=365)
xgb_model.fit(X_train, y_train)
```

## Model Evaluation
The models are evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

A confusion matrix is also generated for both models to visualize performance.

```python
def evaluate_model(y_test, y_pred, model_name):
    print(f"\n--- {model_name} Performance ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.2f}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.2f}")
```

## Credit Worthiness Prediction
You can make predictions using either model with sample input:

```python
predicted_score_xgb = predict_credit_worthiness(xgb_model, 30, 1, 50000, 2, 3, 5, 2, 1, 0, 10, 2, 1, 10000, 0.3, 0, 200, 300, 2, 1000, 120)

predicted_score_rf = predict_credit_worthiness(rf_model, 30, 1, 50000, 2, 3, 5, 2, 1, 0, 10, 2, 1, 10000, 0.3, 0, 200, 300, 2, 1000, 120)
```

## Results
- The **XGBoost** and **Random Forest** models both provide predictions on whether an individual has **Poor, Standard, or Good** credit.
- The model with the best performance can be selected based on evaluation metrics.

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/CodeAlpha_Tasks.git
   cd CodeAlpha_Tasks/Credit Scoring Model
   ```
2. Place your dataset in the `Data/` directory.
3. Run the script:
   ```bash
   python task.py
   ```
4. Check the terminal output for model evaluation and predictions.

## License
This project is open-source and available under the MIT License.
