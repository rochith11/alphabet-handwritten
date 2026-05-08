!pip install tensorflow opencv-python-headless numpy matplotlib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from google.colab import drive

# ---------------------------------------------------------
# SETUP & PATHS
# ---------------------------------------------------------
# Mount Google Drive
drive.mount('/content/drive')

MODEL_DIR = '/content/drive/MyDrive/models'
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, 'alphabet_cnn.h5')

# Define CSV paths (ensure these match your extracted dataset location)
train_csv_path = '/content/dataset/emnist-letters-train.csv'
test_csv_path  = '/content/dataset/emnist-letters-test.csv'

# ---------------------------------------------------------
# PREPROCESSING FUNCTION
# ---------------------------------------------------------
def load_and_preprocess_csv(csv_path):
    """
    1. Loads CSV using Pandas
    2. Separates labels and pixels
    3. Reshapes and corrects EMNIST orientation
    4. Normalizes pixel values to [0, 1]
    5. Adjusts labels to 0-25
    """
    print(f"Loading data from {csv_path}...")
    # 1. Load CSV (EMNIST CSVs usually lack headers)
    data = pd.read_csv(csv_path, header=None)

    # 2. Separate labels (column 0) and pixel values (columns 1-784)
    labels = data.iloc[:, 0].values
    pixels = data.iloc[:, 1:].values

    # 5. Adjust labels from 1-26 into 0-25 (Crucial for Keras 0-indexed categories)
    labels = labels - 1

    # 3. Reshape pixel arrays into 28x28x1
    pixels = pixels.reshape(-1, 28, 28, 1)

    # Correct the transposition issue inherent to EMNIST CSV data
    # (Rotate 90 degrees and flip so letters stand upright)
    pixels = np.array([np.fliplr(np.rot90(img, axes=(1, 0))) for img in pixels])

    # 4. Normalize pixel values
    pixels = pixels.astype('float32') / 255.0

    return pixels, labels

# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------
# 11 & 12. Check for existing model to avoid retraining
if os.path.exists(MODEL_PATH):
    print(f"\n[INFO] Saved model found at {MODEL_PATH}. Loading to skip retraining...")
    model = load_model(MODEL_PATH)

    # We still need the test data for evaluation and prediction
    x_test, y_test = load_and_preprocess_csv(test_csv_path)
    y_test_cat = to_categorical(y_test, num_classes=26)

else:
    print("\n[INFO] No saved model found. Preparing data for training...")

    # Load Train and Test sets
    x_train, y_train = load_and_preprocess_csv(train_csv_path)
    x_test, y_test   = load_and_preprocess_csv(test_csv_path)

    # One-hot encode the 0-25 labels into 26 classes
    y_train_cat = to_categorical(y_train, num_classes=26)
    y_test_cat  = to_categorical(y_test, num_classes=26)

    # 6. Print dataset shapes
    print("\n--- Dataset Shapes ---")
    print(f"Training Data:   {x_train.shape}")
    print(f"Training Labels: {y_train_cat.shape}")
    print(f"Testing Data:    {x_test.shape}")
    print(f"Testing Labels:  {y_test_cat.shape}\n")

    # 7. Build lightweight CNN architecture
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(26, activation='softmax') # 26 classes for A-Z
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # 8. Train the model (Colab uses GPU automatically if enabled in Runtime settings)
    print("\n[INFO] Starting training...")
    history = model.fit(
        x_train,
        y_train_cat,
        epochs=15,
        batch_size=128,
        validation_split=0.1
    )

    # 10. Save the trained model to Google Drive
    model.save(MODEL_PATH)
    print(f"\n[SUCCESS] Model persistently saved to: {MODEL_PATH}")

# ---------------------------------------------------------
# EVALUATION & PREDICTION
# ---------------------------------------------------------
# 9. Evaluate model accuracy
print("\n[INFO] Evaluating model on test data...")
loss, accuracy = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"Final Test Accuracy: {accuracy * 100:.2f}%")

# 13. Show a sample prediction on a test image
sample_idx = 42 # Change this index to see different letters
sample_image = x_test[sample_idx]
true_label = y_test[sample_idx]

# Model expects a batch, so reshape (1, 28, 28, 1)
prediction_probs = model.predict(sample_image.reshape(1, 28, 28, 1), verbose=0)
predicted_label = np.argmax(prediction_probs)

# Convert 0-25 integer back to 'A'-'Z' character
true_char = chr(true_label + ord('A'))
predicted_char = chr(predicted_label + ord('A'))

# Plot the result
plt.figure(figsize=(3, 3))
plt.imshow(sample_image.reshape(28, 28), cmap='gray')
plt.title(f"True: {true_char} | Predicted: {predicted_char}")
plt.axis('off')
plt.show()

model.save("alphabet_model.keras")
from google.colab import files
import os

# Download the .h5 model file from Google Drive
model_h5_path = '/content/drive/MyDrive/model/alphabet_cnn.h5'
if os.path.exists(model_h5_path):
    files.download(model_h5_path)
    print(f"Downloaded {model_h5_path}")
else:
    print(f"Error: .h5 model file not found at {model_h5_path}")

from google.colab import files
import os

# Download the .keras model file saved in the current session
model_keras_path = 'alphabet_model.keras'
if os.path.exists(model_keras_path):
    files.download(model_keras_path)
    print(f"Downloaded {model_keras_path}")
else:
    print(f"Error: .keras model file not found at {model_keras_path}")

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))

for i in range(10):
    plt.subplot(2,5,i+1)

    plt.imshow(x_train[i+1].reshape(28,28), cmap='gray')

    label = chr(np.argmax(y_train_cat[i+1]) + ord('A'))

    plt.title(label)

    plt.axis('off')

plt.show()

import opencv as cv2
cv2.imwrite("test.png", x_test[0] * 255)

