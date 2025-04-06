import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import os

model = load_model("1-94.90-256x256.h5")
labels = ["Kutup Ayısı", "Panda", "Kahverengi Ayı"]

def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif;*.webp")])
    if not file_path:
        return

    img = cv2.imread(file_path)
    imgAI = cv2.resize(img, (256, 256))
    imgAI = np.expand_dims(imgAI, axis=0)

    img = cv2.resize(img, (600, 600), interpolation=cv2.INTER_LANCZOS4)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)

    panel.config(image=img)
    panel.image = img
    prediction = model.predict(imgAI)
    print(prediction)
    class_index = np.argmax(prediction[0])
    label.config(text=f"{labels[class_index]}")


root = tk.Tk()
root.config(background="#2b2b2b")
root.title("Bear Detection")

btn = tk.Button(root, text="Resim Seç", command=open_image)
btn.pack()

label = tk.Label(root,background="#2b2b2b",foreground="#ffffff")
label.pack()

panel = tk.Label(root,background="#2b2b2b")
panel.pack()

root.geometry("600x620")
root.mainloop()