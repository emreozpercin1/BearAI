import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

labels = ["Kutup Ayısı", "Panda", "Kahverengi Ayı"]

model = load_model("1-94.90-256x256.h5")

def predict(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))
    img = np.expand_dims(img, axis=0)

    tahminler = model.predict(img)
    class_index = np.argmax(tahminler)
    class_name = labels[class_index]
    
    persentage = np.max(tahminler)
    return class_name, persentage

for i in os.listdir("testimages"):
    d = os.path.join("testimages", i)
    sinif, olasilik = predict(str(d))
    print("----------------------------------------------------->    "+str(i)+" = "+sinif)