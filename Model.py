import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense ,AveragePooling2D ,Dropout

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

egitim_verileri, dogrulama_verileri, egitim_etiketler, dogrulama_etiketler = train_test_split(egitim_verileri, etiketler, test_size=0.025, random_state=42)

model = Sequential([

    Conv2D(8, (3, 3), activation='relu', input_shape=(256, 256, 3)),

    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(1024, activation='relu'),
    Dense(256, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(egitim_verileri, egitim_etiketler, epochs=80, batch_size=32, validation_data=(dogrulama_verileri, dogrulama_etiketler))

model.save("TestModel.h5")
