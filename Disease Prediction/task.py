import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from google.colab import drive

drive.mount('/content/drive')
dataset_path = "/content/drive/MyDrive/ProjectData/heart.csv"

raw_data = pd.read_csv(dataset_path)
print(raw_data.head(), end = "\n\n")

print("Dataset Shape:", raw_data.shape)
raw_data.info()

print("Missing Values in Each Column:\n", raw_data.isnull().sum())
print(raw_data.describe(), end = "\n\n")

data = raw_data.copy()
print("Target Variable Distribution:\n", data['target'].value_counts(), end = "\n\n")

X = data.drop(columns='target', axis=1)
Y = data['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Initialize the Logistic Regression model with increased max_iter

model = LogisticRegression(max_iter=2000, solver='liblinear')
model.fit(X_train, Y_train)
X_train_prediction = model.predict(X_train)

#Check for training data generalization

training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training Data:', training_data_accuracy, end = "\n\n")

X_test_prediction = model.predict(X_test)

# Calculate for test data

test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on Test Data:', test_data_accuracy, end = "\n\n")

# Dummy data
input_data = np.array([[48, 0, 0, 120, 255, 0, 0, 160, 0, 3.3, 0, 1, 2]])
input_df = pd.DataFrame(input_data, columns=X.columns)

if input_df.shape[1] != X.shape[1]:
    raise ValueError("Input data does not match features used for training")

#New Pred
prediction = model.predict(input_df)

if prediction[0] == 0:
    print('The Person does NOT have Heart Disease')
else:
    print('The Person HAS Heart Disease')
