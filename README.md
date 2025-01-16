# Object Detection Using Simple CNN

## Overview
This project demonstrates the use of a simple Convolutional Neural Network (CNN) for object detection. The model is capable of classifying objects and localizing them by predicting bounding boxes. This implementation serves as a foundational step toward understanding object detection models.

---

## Features
- **Custom CNN Architecture**: A lightweight, efficient model designed for classification and localization tasks.
- **Data Augmentation**: Techniques like random rotation, flips, and color jitter to improve model robustness.
- **Evaluation Metrics**: IoU, mAP, and accuracy for assessing model performance.
- **Comparison with Pre-trained Models**: Benchmarked against YOLOv8 for performance evaluation.

---

## Table of Contents
1. [Background](#background)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Data Preprocessing](#data-preprocessing)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Results](#results)
7. [Setup and Usage](#setup-and-usage)
8. [Future Work](#future-work)
9. [Contributors](#contributors)

---

## Background
Object detection is a critical computer vision task combining:
- **Classification**: Identifying object type.
- **Localization**: Determining object location with bounding boxes.

This project explores the implementation of a simple CNN for object detection and benchmarks it against advanced models like YOLOv8.

---

## Dataset
The dataset used in this project contains annotated images of road signs. Each image includes:
- **Class Labels**: The type of road sign (e.g., "Stop", "Speed Limit").
- **Bounding Boxes**: The coordinates of the objects within the image.

### Data Source
The dataset was sourced from Kaggle and was preloaded in the training environment.

---

## Model Architecture
The custom CNN includes:
1. **Convolutional Layers**: For feature extraction.
2. **Activation Functions**: ReLU for non-linearity.
3. **Pooling Layers**: To reduce spatial dimensions.
4. **Fully Connected Layers**: For classification and bounding box regression.

### Key Features
- Dual outputs for **class predictions** and **bounding box coordinates**.
- Lightweight design for training on moderate hardware.

---

## Data Preprocessing
Several data augmentation techniques were applied to improve model generalization:
1. **Random Rotation**: ±30° to simulate varied orientations.
2. **Horizontal and Vertical Flips**: For orientation invariance.
3. **Color Jittering**: Adjusting brightness, contrast, saturation, and hue.
4. **Resizing**: Scaling images to 128x128 pixels.
5. **Normalization**: Converting image pixel values to tensors.

---

## Evaluation Metrics
1. **Intersection over Union (IoU)**: Measures overlap between predicted and actual bounding boxes.
2. **Mean Average Precision (mAP)**: Combines precision and recall to evaluate detection performance.
3. **Accuracy**: Tracks model performance in predicting the correct class.

---

## Results
### Custom CNN Performance
- **IoU**: Improved from 0.0651 to 0.1996 over 40 epochs.
- **mAP**: Increased from 0.0451 to 0.2520.
- **Accuracy**: Reached ~80% within the first 10 epochs.
  
<img width="356" alt="image" src="https://github.com/user-attachments/assets/eefad893-5890-429d-97eb-6cd4d6f16c9b" />


### YOLOv8 Performance
- **IoU**: Achieved a mean IoU of 0.802.
- **mAP**: Reported at 0.957, significantly outperforming the custom CNN.

---

## Setup and Usage
### Prerequisites
- Python 3.7 or higher
- PyTorch 1.10+
- Required libraries: torchvision, matplotlib, numpy, pandas.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/object-detection-cnn.git
