
#  Transfer Learning with VGG16 — Cats vs Dogs Classification

This project demonstrates how to build an image classifier using **Transfer Learning** with **VGG16** on the Cats vs Dogs dataset.
It includes data preprocessing, model training, evaluation, and heatmap visualization for model interpretability.

## Features
* Uses **TensorFlow** and **Keras** for Transfer Learning
* Preprocessing with `tf` pipelines
* VGG16 as a feature extractor + custom classifier head
* Training with callbacks (EarlyStopping, LR Scheduler, Checkpointing)
* Accuracy & loss visualization
* Grad-CAM heatmaps to interpret predictions
* Supports testing the model on new images

## Performance
* Achieved **~98% validation accuracy**
* Stable training curves with low validation loss


## Requirements
* Python 3.x
* TensorFlow
* tensorflow-datasets
* Matplotlib
* seaborn
* NumPy

## Dataset
You can use either:
* **`image_dataset_from_directory`** (local dataset)
* **`tensorflow_datasets`** (TFDS `cats_vs_dogs`)

