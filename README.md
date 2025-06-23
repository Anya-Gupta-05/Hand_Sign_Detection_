# 🤖 Hand Sign Detection of Alphabets using YOLOv8m

This project focuses on detecting hand signs corresponding to English alphabets using a custom-trained YOLOv8m object detection model. It leverages deep learning to identify and classify hand gestures captured via images.

## 📌 Project Overview

The goal of the project is to detect hand signs representing the English alphabet (A-Z). It involves collecting, annotating, training, and evaluating a YOLOv8m model that can accurately recognize individual hand signs from images.

## 🧠 Approach

1. **Custom Dataset Creation**:
   - A custom dataset of hand signs was created manually.
   - Images were collected and uploaded to [Roboflow](https://roboflow.com/) for annotation.
   - Each image was labeled with its corresponding alphabet using bounding boxes.
   - Roboflow was also used to export the dataset in **YOLOv8 format**, making it ready for model training.

2. **Model Training**:
   - The YOLOv8m model (a medium variant of the Ultralytics YOLOv8 architecture) was trained on the dataset over **20 epochs**.
   - Training involved loss monitoring, precision-recall curve evaluation, and confusion matrix visualization.
   - Training and validation metrics were recorded for performance analysis.

3. **Model Inference**:
   - After training, the best model weights (`best.pt`) were used for inference.
   - The notebook `hand_sign_detection.ipynb` demonstrates how to load the trained model and run predictions on new hand sign images.
## 📓 Notebooks Explained

### 1. `yolov8m_training.ipynb`
- Responsible for loading the dataset, setting up the YOLOv8m training pipeline, and evaluating the model.
- Includes dataset analysis, hyperparameter tuning (epochs, learning rate, etc.), and visualization of training metrics.

### 2. `hand_sign_detection.ipynb`
- Demonstrates how to use the trained `best.pt` weights to perform inference.
- Accepts test images and outputs predictions with bounding boxes and labels for each detected hand sign.


## 📁 Project Assets

### 📦 `Hand_Sign_Detection_dataset.zip`
- Contains the entire **training dataset** in YOLOv8 format.
- Inside the zip:
  - `images/`: Training and validation images.
  - `labels/`: Corresponding YOLO-format annotation files.
  - `data.yaml`: Defines the dataset splits and class mappings for YOLOv8.

### 📊 `metrics/`
This directory includes various visualizations and analysis outputs from training:
- `precision-recall_curve.png`: Visualizes the trade-off between precision and recall.
- `recall_curve.png`: Shows recall variation per class.
- `confusion_matrix.png`: Indicates model performance across all classes.
- `labels.jpg`: Visualization of bounding boxes across the dataset.
- `F1_curve.png`: F1 score trends across confidence thresholds.

### 📈 `results/`
- `results.csv`: Epoch-wise training log that includes loss values, precision, recall, mAP@0.5, mAP@0.5:0.95, and learning rate.
- `results.png`: A plot showing training and validation losses, along with performance metrics over time.

## 📊 Results

| Metric              | Value    |
|---------------------|----------|
| **Precision**       | ~87%     |
| **Recall**          | ~93%     |
| **mAP@0.5**         | ~92%     |
| **mAP@0.5:0.95**    | ~58%     |

The model performs well at detecting and classifying hand signs with high precision and recall, making it suitable for real-world applications such as gesture-based communication tools.


