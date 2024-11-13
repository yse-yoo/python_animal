import numpy as np
import cv2
from keras.models import load_model
import os
import tkinter as tk
from PIL import ImageTk, Image
from categories import categories  # クラス名をキーに持つ categories をインポート

# 確率のしきい値
THRESHOLD = 0.7
# モデル入力用画像サイズ
IMG_SIZE = 64  
# GUI表示用の画像サイズ
DISPLAY_SIZE = 128  

# 現在のスクリプトファイルのディレクトリを取得
base_dir = os.path.dirname(os.path.abspath(__file__))

# モデルを読み込む
model_path = os.path.join(base_dir, 'bird_classifier_model.h5')
model = load_model(model_path)

# 画像ディレクトリのパス
images_dir = os.path.join(base_dir, 'images', 'notraining')

# 画像ファイルリストを取得
image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# インデックスとクラス名をマッピングする辞書を作成
label_map = {idx: category for idx, category in enumerate(categories.keys())}

# GUIのセットアップ
root = tk.Tk()
root.title("Bird Classifier")
root.geometry("600x600")

img_label = tk.Label(root)
img_label.pack()

result_label = tk.Label(root, font=("Helvetica", 18))
result_label.pack()

# 画像インデックスの初期値
current_image_index = 0

# 予測と表示を行う関数
def predict_and_display():
    global current_image_index
    
    if current_image_index >= len(image_files):
        result_label.configure(text="No more images.")
        return
    
    file_path = os.path.join(images_dir, image_files[current_image_index])
    image = cv2.imread(file_path)
    if image is None:
        result_label.configure(text="Image not found or cannot be read.")
        return
    
    # 画像を前処理
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGBに変換
    image_resized = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))  # モデル用サイズにリサイズ
    image_normalized = np.expand_dims(image_resized, axis=0) / 255.0   # 正規化と次元追加

    # 予測を実行
    prediction = model.predict(image_normalized)
    class_idx = np.argmax(prediction)
    confidence = prediction[0][class_idx]  # 最も高い予測確率

    # 確率がしきい値以下の場合は「その他」として表示
    if confidence < THRESHOLD:
        result_text = "その他"
    else:
        # インデックスからクラス名を取得し、カテゴリ情報からラベルを取得
        class_name = label_map.get(class_idx, "Unknown")
        result_text = categories.get(class_name, {}).get("label", "Unknown class")
    
    result_label.configure(text=f"確率: {confidence:.2f} - {result_text}")

    # GUI表示用に画像をリサイズ
    display_image = Image.fromarray(image_rgb).resize((DISPLAY_SIZE, DISPLAY_SIZE))
    img_tk = ImageTk.PhotoImage(display_image)
    img_label.configure(image=img_tk)
    img_label.image = img_tk

    # 次の画像に進む
    current_image_index += 1

# 「次の画像」ボタンを作成
next_button = tk.Button(root, text="Next Image", command=predict_and_display)
next_button.pack()

# 最初の画像を表示
predict_and_display()

root.mainloop()