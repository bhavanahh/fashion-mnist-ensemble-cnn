import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Fashion-MNIST dataset
(x_train_all, y_train_all), (x_test_all, y_test_all) = keras.datasets.fashion_mnist.load_data()

# Select first 50 training and testing records
x_train = x_train_all[:50]
y_train = y_train_all[:50]

x_test = x_test_all[:50]
y_test = y_test_all[:50]

print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)

# Normalize pixel values between 0 and 1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("Minimum pixel value:", x_train.min())
print("Maximum pixel value:", x_train.max())

# Reshape images for CNN
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print("Training shape after reshaping:", x_train.shape)
print("Testing shape after reshaping:", x_test.shape)

# Split training data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    y_train,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(x_train))
print("Validation samples:", len(x_val))

# Function to build CNN model
def build_cnn():

    model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(32, (3, 3), activation="relu"),

        # Max pooling layer
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten layer
        layers.Flatten(),

        # Output layer
        layers.Dense(10, activation="softmax")
    ])

    # Compile the model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# Create a CNN model
model = build_cnn()

# Display model architecture
model.summary()


# Train 5 CNN models using Bootstrap Sampling

ensemble_models = []

for i in range(5):

    print(f"\nTraining CNN Model {i + 1}")

    # Generate bootstrap sample
    indices = np.random.choice(
        len(x_train),
        size=len(x_train),
        replace=True
    )

    x_bootstrap = x_train[indices]
    y_bootstrap = y_train[indices]

    # Build a new CNN model
    model = build_cnn()

    # Train the model
    model.fit(
        x_bootstrap,
        y_bootstrap,
        epochs=3,
        batch_size=8,
        verbose=1
    )

    # Store the trained model
    ensemble_models.append(model)

print("\nAll 5 CNN models trained successfully!")

# Ensemble prediction on validation data

val_predictions = []

for model in ensemble_models:
    predictions = model.predict(x_val, verbose=0)
    val_predictions.append(predictions)

# Average probabilities of all 5 models
ensemble_val_prob = np.mean(val_predictions, axis=0)

# Select class with highest probability
ensemble_val_pred = np.argmax(ensemble_val_prob, axis=1)

# Calculate validation accuracy
ensemble_val_accuracy = accuracy_score(y_val, ensemble_val_pred)


# Ensemble prediction on test data

test_predictions = []

for model in ensemble_models:
    predictions = model.predict(x_test, verbose=0)
    test_predictions.append(predictions)

# Average probabilities
ensemble_test_prob = np.mean(test_predictions, axis=0)

# Final predictions
ensemble_test_pred = np.argmax(ensemble_test_prob, axis=1)

# Calculate test accuracy
ensemble_test_accuracy = accuracy_score(y_test, ensemble_test_pred)


print("\nENSEMBLE CNN RESULTS")
print("Validation Accuracy:", ensemble_val_accuracy)
print("Test Accuracy:", ensemble_test_accuracy)

# Build a single CNN model
single_model = build_cnn()

# Train the single CNN
print("\nTraining Single CNN Model")

single_model.fit(
    x_train,
    y_train,
    epochs=3,
    batch_size=8,
    verbose=1
)

# Validation prediction
single_val_prob = single_model.predict(x_val, verbose=0)
single_val_pred = np.argmax(single_val_prob, axis=1)

single_val_accuracy = accuracy_score(y_val, single_val_pred)

# Test prediction
single_test_prob = single_model.predict(x_test, verbose=0)
single_test_pred = np.argmax(single_test_prob, axis=1)

single_test_accuracy = accuracy_score(y_test, single_test_pred)


# Display comparison
print("\nSINGLE CNN RESULTS")
print("Validation Accuracy:", single_val_accuracy)
print("Test Accuracy:", single_test_accuracy)


# Final comparison
print("\nFINAL MODEL COMPARISON")
print("--------------------------------")
print("Model       Validation     Test")
print("--------------------------------")
print("Ensemble    ", ensemble_val_accuracy, "   ", ensemble_test_accuracy)
print("Single CNN  ", single_val_accuracy, "   ", single_test_accuracy)