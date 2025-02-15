readme_content = """
# Heart Disease Prediction Model

## Overview
This project uses a **Logistic Regression** model to predict whether a person has heart disease based on various medical attributes.

## Dataset
The dataset used is stored at **Data/heart.csv**. It contains multiple medical attributes such as age, cholesterol levels, and blood pressure, along with a target variable (`target`):
- `0` - No heart disease
- `1` - Has heart disease

## Dependencies
The following Python libraries are required:
```bash
pip install numpy pandas scikit-learn
```

## Usage
### 1. Load and Explore Data
```python
import pandas as pd

dataset_path = "Data/heart.csv"
raw_data = pd.read_csv(dataset_path)
print(raw_data.head())
print(raw_data.info())
```
### 2. Train Model
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X = raw_data.drop(columns='target')
Y = raw_data['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

model = LogisticRegression(max_iter=2000, solver='liblinear')
model.fit(X_train, Y_train)
```

### 3. Evaluate Model
```python
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on Test Data:', test_data_accuracy)
```

### 4. Make Predictions
```python
import numpy as np

input_data = np.array([[48, 0, 0, 120, 255, 0, 0, 160, 0, 3.3, 0, 1, 2]])
input_df = pd.DataFrame(input_data, columns=X.columns)
prediction = model.predict(input_df)

if prediction[0] == 0:
    print('The Person does NOT have Heart Disease')
else:
    print('The Person HAS Heart Disease')
```

## License
This project is open-source and available under the MIT License.
"""

with open("README.md", "w") as f:
    f.write(readme_content)

print("README.md file has been created successfully.")

