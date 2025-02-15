# Handwriting Recognition with EMNIST and IAM Dataset

## Overview
This project builds and trains a CNN-based handwriting recognition model using the EMNIST dataset for character recognition and the IAM dataset for word recognition.

## Datasets Used
- **EMNIST ByClass**: Used for recognizing handwritten characters (letters and digits).
- **IAM Handwriting Database**: Used for recognizing full handwritten words.

## Project Structure
```
/
├── Data/                      # Contains the EMNIST dataset (MAT file)
├── handwriting_data/          # Extracted IAM dataset
│   ├── samples/               # Handwriting image samples
│   ├── words.txt              # Metadata for IAM dataset
├── model/                     # Saved model (optional)
├── main.py                    # Main script to load datasets and train models
└── README.md                  # Project documentation
```

## Dependencies
Make sure you have the following dependencies installed:
```bash
pip install tensorflow numpy matplotlib scipy
```

## Model Training
The CNN model is trained using the EMNIST dataset for character recognition with:
- **3 Convolutional Layers** with ReLU activation
- **Batch Normalization and MaxPooling**
- **Dense layers** with dropout to prevent overfitting
- **Adam optimizer** with categorical cross-entropy loss

Training command:
```python
python main.py
```

## Results & Evaluation
The trained model is evaluated on:
- Accuracy and loss over epochs
- Predictions on sample test data
- Visualization of actual vs. predicted characters

## Future Improvements
- Implement LSTM/Transformer models for IAM dataset word recognition.
- Improve data augmentation for better generalization.
- Experiment with different CNN architectures.

## Credits
Dataset sources:
- EMNIST: https://www.nist.gov/itl/products-and-services/emnist-dataset
- IAM Handwriting: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database

---
### Author
*Your Name Here*
