# 🍛 Indian Food Classifier using Deep Learning

A deep learning-based image classification project that identifies Indian food dishes using **MobileNetV2 (Transfer Learning)** and provides a **Tkinter GUI application** for real-time predictions.

## 📌 Project Overview
This project classifies Indian food images into different categories using a CNN-based model. It leverages **MobileNetV2 pretrained on ImageNet** for better accuracy and faster training. A simple desktop GUI allows users to upload images and get instant predictions.

## 🚀 Features
- 🍽️ Classifies Indian food images
- 🤖 Transfer Learning with MobileNetV2
- 🧠 High accuracy with CNN architecture
- 💾 Automatic model saving & loading
- 🖥️ GUI-based image prediction using Tkinter
- ⚡ Real-time prediction on uploaded images

## ⚙️ Tech Stack
- Python 🐍
- TensorFlow / Keras 🤖
- NumPy 🔢
- Pillow (PIL) 🖼️
- Tkinter 🖥️
- MobileNetV2 (Transfer Learning)

## 🧠 Model Architecture
- Base Model: MobileNetV2 (ImageNet weights)
- GlobalAveragePooling2D
- Dense Layer (128 neurons, ReLU)
- Dropout (0.3)
- Output Layer (Softmax activation)

## 📊 Workflow
1. Load dataset using ImageDataGenerator
2. Preprocess images (resize + rescale)
3. Build model using MobileNetV2
4. Train model (if not already saved)
5. Save model as `.keras`
6. Load model for predictions
7. Predict food class from uploaded image
8. Display result in GUI
