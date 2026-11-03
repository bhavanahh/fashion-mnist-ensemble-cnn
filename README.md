# Ensemble CNN for Fashion-MNIST Classification

## Overview

This project implements an Ensemble Convolutional Neural Network (CNN) to classify Fashion-MNIST clothing images.

Five CNN models are trained using Bootstrap Sampling. Their predictions are combined using Average Probability Voting.

The performance of the ensemble model is compared with a single CNN model.

## Dataset

Fashion-MNIST dataset is loaded using `keras.datasets.fashion_mnist`.

- Training images selected: First 50 records
- Testing images selected: First 50 records
- Image dimensions: 28 × 28 pixels
- Number of classes: 10
- Image type: Grayscale

## Data Preprocessing

The following preprocessing steps are performed:

1. Selected the first 50 training and testing images.
2. Normalized pixel values from 0–255 to 0–1.
3. Reshaped images from (28, 28) to (28, 28, 1).
4. Split the training data into training and validation sets.
   - Training samples: 40
   - Validation samples: 10

## CNN Architecture

Each CNN model uses the following architecture:

| Layer | Configuration |
|---|---|
| Conv2D | 32 filters, 3 × 3 kernel, ReLU |
| MaxPooling2D | 2 × 2 pool size |
| Flatten | Converts feature maps into a 1D vector |
| Dense | 10 neurons, Softmax activation |

### Model Compilation

- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Evaluation Metric: Accuracy
- Epochs: 3
- Batch Size: 8

## Ensemble Learning

Five independent CNN models are trained using Bootstrap Sampling.

Each bootstrap dataset contains 40 randomly selected training samples, with replacement.

The predictions from all five models are combined by averaging their class probabilities.

The class with the highest average probability is selected as the final prediction.

## Single CNN Model

A separate CNN with the same architecture is trained using the complete training split of 40 samples.

Its validation and testing accuracy are compared with the ensemble model.

## Results

The following results were obtained during the experiment:

| Model | Validation Accuracy | Test Accuracy |
|---|---:|---:|
| Ensemble CNN | 40% | 52% |
| Single CNN | 50% | 48% |

### Observation

- The Single CNN achieved higher validation accuracy.
- The Ensemble CNN achieved higher test accuracy.
- The ensemble improved test accuracy by 4 percentage points compared with the single CNN.

The experiment uses only 50 training and 50 testing images, so the results may vary between runs.

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Scikit-learn
- Matplotlib

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python ensemble_cnn.py
```

## Project Structure

```text
fashion-mnist-ensemble/
│
├── ensemble_cnn.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Conclusion

This project demonstrates how Ensemble Learning can combine multiple CNN models to perform image classification.

Five CNN models are trained using Bootstrap Sampling, and their predictions are combined using Average Probability Voting.

The ensemble model achieved 52% test accuracy, compared with 48% for the single CNN in this experiment.git init
git 