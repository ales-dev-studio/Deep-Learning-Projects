
# MNIST Handwritten Digits Recognizer

## Project Overview

The **MNIST Handwritten Digits Recognizer** implemented with both Scikit-learn and TensorFlow.
Two complete neural network approaches for handwritten digit recognition. Includes data preprocessing, model training, and comprehensive evaluation with performance visualization.

---

## First notebook using scikit-learn

### Technologies & Libraries Used

| Category                     | Libraries                                                     |
| ---------------------------- | ------------------------------------------------------------- |
| **Core ML / Neural Network** | `scikit-learn` (`MLPClassifier`)                              |
| **Data Handling**            | `numpy`, `pandas`                                             |
| **Visualization**            | `matplotlib`, `seaborn`                                       |
| **Dataset Access**           | `fetch_openml` from `sklearn.datasets`                        |
| **Preprocessing**            | `StandardScaler`                                              |
| **Evaluation**               | `accuracy_score`, `classification_report`, `confusion_matrix` |


### Results Summary

| Metric                | Training                                             | Testing |
| --------------------- | ---------------------------------------------------- | ------- |
| **Accuracy**          | ~99%                                                 | ~97–98% |
| **Best Performance**  | On digits with clear shape (e.g., 0, 1, 7)           |         |
| **Common Confusions** | Between visually similar digits (e.g., 3 & 5, 4 & 9) |         |

The MLPClassifier achieves high accuracy without deep learning frameworks, demonstrating that even a simple feedforward neural network can perform well on structured datasets like MNIST.

---

### Key Learnings

* Importance of feature scaling in neural networks.
* Practical workflow for supervised image classification using scikit-learn.
* Visualization techniques for understanding model behavior.
* Basics of MLP architecture tuning.

---

### Possible Improvements

* Increase training iterations (`max_iter=300` or `early_stopping=True`).
* Try different architectures (e.g., more layers or neurons).
* Use **Convolutional Neural Networks (CNNs)** via TensorFlow/Keras for better accuracy.
* Apply **Dimensionality Reduction (PCA)** to speed up training.

---

### Project Files

| File                                          | Description                                                          |
| --------------------------------------------- | -------------------------------------------------------------------- |
| **MNIST-Handwritten-Digits-Recognizor.ipynb** | Main Jupyter notebook for training and evaluating the MLP classifier |
| *(Optional)* `requirements.txt`               | Python dependencies                                                  |
| *(Optional)* `README.md`                      | Project documentation (this file)                                    |

---

## Second notebook using TensorFlow

## Project Overview

This notebook implements a comprehensive deep learning solution for handwritten digit classification using TensorFlow on the MNIST dataset. The implementation demonstrates professional-grade neural network architecture with advanced techniques to achieve state-of-the-art performance in digit recognition.

## Model Architecture

**Deep Neural Network with Advanced Components:**
```python
# Multi-layer architecture with regularization
Input (784) → Dense(512) → BatchNorm → Dropout(0.3)
           → Dense(256) → BatchNorm → Dropout(0.3) 
           → Dense(128) → BatchNorm → Dropout(0.2)
           → Dense(64) → BatchNorm → Dropout(0.2)
           → Output(10, softmax)
```

**Key Features:**
- **Multiple Hidden Layers**: 512 → 256 → 128 → 64 neurons for hierarchical feature learning
- **Batch Normalization**: Stabilizes training and accelerates convergence
- **Dropout Regularization**: Prevents overfitting (30% dropout rates)
- **Advanced Activation**: ReLU activations for better gradient flow
- **Professional Training**: Adam optimizer with learning rate scheduling