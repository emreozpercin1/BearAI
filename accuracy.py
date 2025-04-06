import os
import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Veri yükleme ve işleme
ana_dizin_yolu = "data"
labels = ["PolarBear", "Panda", "BrownBear"]

egitim_verileri = []
etiketler = []

for sinif_index, sinif in enumerate(labels):
    klasor_yolu = os.path.join(ana_dizin_yolu, sinif)
    
    for dosya in os.listdir(klasor_yolu):
        dosya_yolu = os.path.join(klasor_yolu, dosya)
        
        foto = cv2.imread(dosya_yolu)
        foto = cv2.resize(foto, (256, 256))

        egitim_verileri.append(foto)

        etiket = [0] * len(labels)
        etiket[sinif_index] = 1
        etiketler.append(etiket)

egitim_verileri = np.array(egitim_verileri)
etiketler = np.array(etiketler)

# Verileri eğitim ve doğrulama olarak ayırma
egitim_verileri, dogrulama_verileri, egitim_etiketler, dogrulama_etiketler = train_test_split(egitim_verileri, etiketler, test_size=0.025, random_state=42)

# Modeli yükleme
model = load_model("2-91.84-256x256.h5")

# Eğitim ve doğrulama verileri üzerindeki doğruluğu ölçme
train_loss, train_accuracy = model.evaluate(egitim_verileri, egitim_etiketler, verbose=0)
print(f'Training Accuracy: {train_accuracy * 100:.2f}%')

val_loss, val_accuracy = model.evaluate(dogrulama_verileri, dogrulama_etiketler, verbose=0)
print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')
