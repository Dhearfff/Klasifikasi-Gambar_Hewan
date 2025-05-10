# Klasifikasi-Gambar_Hewan

## 📝 Deskripsi Proyek
Proyek ini bertujuan untuk **mengklasifikasikan gambar hewan** menggunakan model **Convolutional Neural Network (CNN)**. Model telah dikonversi ke format **TF-Lite, TFJS, dan SavedModel** agar dapat digunakan dalam berbagai platform.

## Setup Environment - Terminal
```
python -m venv myenv1

myenv\Scripts\activate

pip install -r requirements.txt
```

## Konversi Model
# Konversi ke SavedModel
```
Model_1.save("saved_model/")
```

# Konversi ke TF-Lite
```
converter = tf.lite.TFLiteConverter.from_keras_model(Model_1)
tflite_model = converter.convert()
with open("model_mobile.tflite", "wb") as f:
    f.write(tflite_model)
```

# Konversi ke TensorFlow.js
```
tfjs.converters.save_keras_model(Model_1, "tfjs_model/")
```

## Project-folder/
│── animal-dataset/         # Dataset gambar hewan
│── dataset-final/          # Dataset final gambar hewan untuk model
│── saved_model/            # Model yang telah dikonversi ke SavedModel
│── tfjs_model/             # Model dalam format TensorFlow.js
│── model_mobile.tflite     # Model dalam format TF-Lite
│── requirements.txt        # Daftar library yang digunakan
│── README.md               # Dokumentasi proyek

