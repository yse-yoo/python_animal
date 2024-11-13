import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Input
from keras.utils import to_categorical
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
from categories import categories

# トレーニングデータの割合設定 (例: 80%をトレーニング用、20%を検証用)
TRAIN_RATIO = 0.8

# Image size for resizing
IMG_SIZE = 64

# 現在のスクリプトファイルのディレクトリを取得
base_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(base_dir, "images")

# カテゴリ情報の辞書
categories = {
    "crow": {"name": "crow", "label": "カラス"},
    "hen": {"name": "hen", "label": "ニワトリ"},
    "stork": {"name": "stork", "label": "コウノトリ"},
    # "other": {"name": "other", "label": "その他"}
}

# カテゴリ名とインデックスをマッピング
label_map = {name: idx for idx, name in enumerate(categories)}

# 画像とラベルを読み込む関数
def load_data():
    data = []
    labels = []
    for category, info in categories.items():
        path = os.path.join(images_dir, info["name"])  # name を使ってフォルダパスを生成
        if not os.path.exists(path):
            raise FileNotFoundError(f"Directory '{path}' does not exist. Please create it and add images.")
        
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB format
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    data.append(img)
                    labels.append(label_map[category])  # インデックスをラベルとして追加

    data = np.array(data, dtype=np.float32) / 255.0  # Normalize and convert to float32
    labels = to_categorical(labels, num_classes=len(categories))  # One-hot encode labels
    return data, labels

# Load the data and labels
data, labels = load_data()

# Shuffle data
idx = np.arange(data.shape[0])
np.random.shuffle(idx)
data = data[idx]
labels = labels[idx]

# Split data into training and validation sets based on TRAIN_RATIO
num_samples = len(data)
num_train = int(num_samples * TRAIN_RATIO)
x_train = data[:num_train]
y_train = labels[:num_train]
x_val = data[num_train:]
y_val = labels[num_train:]

# Build the model with an Input layer
model = Sequential()
model.add(Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(len(categories), activation='softmax'))  # Change output layer to match categories

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))

# Save Model
model.save(os.path.join(base_dir, 'bird_classifier_model.h5'))

# Function to handle the image prediction
def predict():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    # Display selected image
    image_tk = Image.open(file_path).resize((IMG_SIZE, IMG_SIZE))
    image_tk = ImageTk.PhotoImage(image_tk)
    img_label.configure(image=image_tk)
    img_label.image = image_tk

    # Prepare the image for prediction
    image_selected = cv2.imread(file_path)
    if image_selected is not None:
        image_selected = cv2.cvtColor(image_selected, cv2.COLOR_BGR2RGB)  # Convert to RGB format
        image_selected = cv2.resize(image_selected, (IMG_SIZE, IMG_SIZE))
        image_selected = np.expand_dims(image_selected, axis=0) / 255.0  # Normalize

        # Predict and display the result
        prediction = model.predict(image_selected)
        class_idx = np.argmax(prediction)
        category_key = list(categories.keys())[class_idx]
        result_text = f"これは{categories[category_key]['label']}です!"
        result_label.configure(text=result_text)

# Create the GUI using tkinter
# root = tk.Tk()
# root.title("Animal Classifier")
# root.geometry("500x500")

# Create GUI widgets
# img_label = tk.Label(root)
# img_label.pack()

# result_label = tk.Label(root, font=("Helvetica", 18))
# result_label.pack()

# button = tk.Button(root, text="Choose Image", command=predict)
# button.pack()

# root.mainloop()