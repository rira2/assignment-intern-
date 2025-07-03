# Digit Recognition with CNN (DevifyX Internship Assignment)

This repository contains my internship assignment for DevifyX. It is a complete, end-to-end handwritten digit recognition project using the MNIST dataset, built with TensorFlow and Keras. The solution meets all the mandatory requirements and includes multiple bonus features to demonstrate model robustness and interpretability.

---

## Project Description

The goal of this project is to develop a robust Convolutional Neural Network (CNN) model that can classify handwritten digits from the MNIST dataset. The notebook walks through data loading, preprocessing, augmentation, model building, hyperparameter tuning, training with early stopping and checkpoints, evaluation, visualization, and deployment-ready model conversion.

**Key steps in the notebook:**
- Load and visualize the MNIST dataset.
- Normalize and reshape the data.
- Apply data augmentation using Keras’ `ImageDataGenerator`.
- Build the CNN architecture with Batch Normalization and Dropout for regularization.
- Tune hyperparameters using Keras Tuner’s Random Search.
- Implement EarlyStopping and ModelCheckpoint callbacks to avoid overfitting.
- Train the final CNN model with optimal hyperparameters.
- Visualize training/validation accuracy and loss curves.
- Generate and display a confusion matrix.
- Identify misclassified samples for manual inspection.
- Save the final model in both `.h5` and `.tflite` formats for deployment.
- Implement inference on a single image.
- Visualize intermediate activations to interpret learned features.
- Demonstrate model robustness by performing the FGSM adversarial attack.
- Improve adversarial robustness through simple adversarial training.

---

## Results

- **Clean test accuracy:** ~98%
- **Adversarial (FGSM) test accuracy before defense:** ~56%
- **Adversarial (FGSM) test accuracy after defense:** ~72% (using adversarial training on a sample subset)

These results demonstrate that while the model performs very well on clean images, small adversarial perturbations can significantly affect performance. By adding adversarial examples to the training set, the model’s robustness improves.

---

---


## Bonus Features Implemented

As per the assignment’s “Bonus” section, I have implemented and demonstrated the following:
- **Hyperparameter tuning** with Keras Tuner.
- **Early stopping and model checkpointing** to prevent overfitting.

- **Defense against adversarial examples (FGSM)** by:
  - Performing the FGSM attack to show model vulnerability.
  - Improving model robustness through simple adversarial training.

---

## How to Run

To run this project on your local machine or Google Colab:

1. **Clone this repository**

   ```bash
   git clone https://github.com/yourusername/digit-recognition-assignment.git
   cd digit-recognition-assignment
