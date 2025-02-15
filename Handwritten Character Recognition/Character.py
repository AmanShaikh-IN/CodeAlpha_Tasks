import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import string

mat_file_path = '/Data/emnist-byclass.mat'

emnist_data = sio.loadmat(mat_file_path)

#Training images and labels
X_train = emnist_data['dataset'][0][0][0][0][0][0]
y_train = emnist_data['dataset'][0][0][0][0][0][1]

#Testing images and labels
X_test = emnist_data['dataset'][0][0][1][0][0][0]
y_test = emnist_data['dataset'][0][0][1][0][0][1]

#Print the shapes of training and testing datasets
print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_test shape: {y_test.shape}')

# Reshape images to (num_samples, 28, 28)
X_train = X_train.reshape(-1, 28, 28)
X_test = X_test.reshape(-1, 28, 28)

# Normalize pixel values to [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Transpose the images to correct orientation
X_train = np.transpose(X_train, (0, 2, 1))
X_test = np.transpose(X_test, (0, 2, 1))

# Flip the images horizontally
X_train = np.flip(X_train, axis=2)
X_test = np.flip(X_test, axis=2)

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

y_train = y_train.flatten().astype(np.int64)
y_test = y_test.flatten().astype(np.int64)
num_classes = len(np.unique(y_train))

#One-hot encoding
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)
emnist_labels = {}

#Digits 0-9
for i in range(10):
    emnist_labels[i] = str(i)

#Uppercase letters
uppercase_letters = list(string.ascii_uppercase)
for i in range(10, 36):
    emnist_labels[i] = uppercase_letters[i - 10]

#Lowercase letters
lowercase_letters = list(string.ascii_lowercase)
for i in range(36, 62):
    emnist_labels[i] = lowercase_letters[i - 36]

#CNN model
model = models.Sequential([
    
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(num_classes, activation='softmax')
])

optimizer = Adam(learning_rate=0.001)

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#Training
history = model.fit(X_train, y_train_cat,
                    validation_data=(X_test, y_test_cat),
                    epochs=15,
                    batch_size=256)

#Evaluation
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=2)
print(f'Test accuracy: {test_acc:.4f}')

#Plotting
plt.figure(figsize=(12, 4))

#Training & Validation Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

#Training & Validation Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Function to predict and display the label for a given sample index

def predict_and_display(sample_index):
    image = X_test[sample_index]
    true_label = y_test[sample_index]

    prediction = model.predict(np.expand_dims(image, axis=0))
    predicted_label = np.argmax(prediction)

    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.title(f'Predicted: {emnist_labels[predicted_label]}, Actual: {emnist_labels[true_label]}')
    plt.axis('off')
    plt.show()

    print(f'Predicted Label: {emnist_labels[predicted_label]}, Actual Label: {emnist_labels[true_label]}')

sample_indices = [10, 200, 600, 1200, 3200]
for idx in sample_indices:
    predict_and_display(idx)
