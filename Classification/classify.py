import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
import tkinter as tk
import os

# =========================
# CONFIG
# =========================
IMG_SIZE = 224
THRESHOLD = 0.5
MODS = ["flair.png", "t1.png", "t1ce.png", "t2.png"]

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model("model.keras", compile=False)
print("✅ Model loaded (CPU mode)")

# =========================
# LOAD 4 MODALITIES
# =========================
imgs = []

for m in MODS:
    if not os.path.exists(m):
        raise FileNotFoundError(f"Missing modality: {m}")

    img = Image.open(m).convert("L").resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img, dtype=np.float32) / 255.0
    imgs.append(img)

x = np.stack(imgs, axis=-1)
x = np.expand_dims(x, axis=0)

# =========================
# INFERENCE
# =========================
prob = float(model.predict(x, verbose=0)[0][0])
pred = "HGG" if prob >= THRESHOLD else "LGG"
confidence = "High" if abs(prob - 0.5) > 0.3 else "Moderate"

# =========================
# GUI WINDOW
# =========================
root = tk.Tk()
root.title("Classification Result")
root.geometry("420x520")
root.resizable(False, False)

# Title
title = tk.Label(
    root,
    text="CLASSIFICATION RESULT",
    font=("Arial", 16, "bold")
)
title.pack(pady=10)

# Input image label
img_label = tk.Label(root, text="Input Image", font=("Arial", 12))
img_label.pack()

# Load and show T1 image
t1_img = Image.open("t1.png").resize((256, 256))
t1_img = ImageTk.PhotoImage(t1_img)

panel = tk.Label(root, image=t1_img)
panel.image = t1_img
panel.pack(pady=10)

# Results text
result_text = (
    f"Predicted Grade : {pred}\n"
    f"Probability     : {prob:.4f}\n"
    f"Confidence      : {confidence}"
)

result_label = tk.Label(
    root,
    text=result_text,
    font=("Arial", 12),
    justify="left"
)
result_label.pack(pady=15)

root.mainloop()
