# ==========================================
# 🍽️ INDIAN FOOD CLASSIFIER PROJECT (FIXED)
# ==========================================

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog

# ==========================================
# 📂 DATASET PATH (CHANGE THIS)
# ==========================================
DATASET_PATH = "Indian Food Images"   # ✅ your dataset folder
MODEL_PATH = "indian_food_model.keras"

# ==========================================
# 🧠 LOAD / CREATE CLASSES
# ==========================================
classes = sorted(os.listdir(DATASET_PATH))
np.save("classes.npy", classes)

# ==========================================
# 🧠 DATA PREPROCESSING
# ==========================================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# ==========================================
# 🧠 MODEL BUILD (MobileNetV2)
# ==========================================
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(len(classes), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================================
# 🚀 TRAIN MODEL (ONLY IF NOT EXISTS)
# ==========================================
if not os.path.exists(MODEL_PATH):
    print("🚀 Training model...")

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=10
    )

    model.save(MODEL_PATH)
    print("✅ Model saved!")

else:
    print("✅ Loading saved model...")
    model = tf.keras.models.load_model(MODEL_PATH)

classes = np.load("classes.npy", allow_pickle=True)

# ==========================================
# 🎯 PREDICTION FUNCTION
# ==========================================
def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224,224))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    return classes[class_index]

# ==========================================
# 🖥️ GUI APP
# ==========================================
def upload_image():
    file_path = filedialog.askopenfilename()

    if not file_path:
        return

    img = Image.open(file_path)
    img = img.resize((250,250))
    img = ImageTk.PhotoImage(img)

    panel.config(image=img)
    panel.image = img

    result = predict_image(file_path)
    label.config(text=f"🍽️ Prediction: {result}")

root = tk.Tk()
root.title("Indian Food Classifier 🍛")
root.geometry("400x500")

btn = tk.Button(root, text="Upload Image", command=upload_image)
btn.pack(pady=20)

panel = tk.Label(root)
panel.pack()

label = tk.Label(root, text="Prediction will appear here", font=("Arial", 14))
label.pack(pady=20)

root.mainloop()