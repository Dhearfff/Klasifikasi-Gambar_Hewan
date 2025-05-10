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
save_path = 'saved_model/'
tf.saved_model.save(model_1, save_path)
print(f"✅ Model berhasil disimpan di {save_path}")

```

# Konversi ke TF-Lite
```
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model/')
tflite_model = converter.convert()

os.makedirs('tflite', exist_ok=True)

with open('tflite/model.tflite', 'wb') as f:
    f.write(tflite_model)

# Simpan label
labels = ["bears", "elephant", "crows", "rats"]
with open('tflite/label.txt', 'w') as f:
    for label in labels:
        f.write(label + '\n')
```

# Konversi ke TensorFlow.js
```
!pip install tensorflowjs
```
```
!tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_format=tfjs_graph_model \
    --signature_name=serving_default \
    --saved_model_tags=serve \
    saved_model/ tfjs_model/
```

# Contoh Inferensi dengan Model SavedModel
```
import tensorflow as tf
import numpy as np
import cv2

# Load model SavedModel
model = tf.keras.models.load_model("saved_model/")

# Baca gambar uji
image_path = "dataset-final/test/bears/2Q__.jpeg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (150, 150))
image = np.expand_dims(image, axis=[0, -1])
image = image.astype(np.float32) / 255.0  

# Melakukan prediksi
prediction = model.predict(image)
predicted_class = np.argmax(prediction)

class_labels = ["Bears", "Elephants", "Crows", "Rats"]
print(f"🔎 Model memprediksi gambar sebagai: {class_labels[predicted_class]}")

```

# Contoh Inferensi dengan Model TF-Lite
```
import tensorflow as tf
import numpy as np
import cv2

# Load model TF-Lite
interpreter = tf.lite.Interpreter(model_path="tflite/model.tflite")
interpreter.allocate_tensors()

# Baca gambar uji dan lakukan preprocessing
image_path = "dataset-final/test/bears/2Q__.jpeg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (150, 150))
image = np.expand_dims(image, axis=[0, -1])
image = image.astype(np.float32) / 255.0  

# Melakukan inferensi
interpreter.set_tensor(interpreter.get_input_details()[0]['index'], image)
interpreter.invoke()
output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

predicted_class = np.argmax(output_data)
class_labels = ["Bears", "Elephants", "Crows", "Rats"]
print(f"🔎 Model memprediksi gambar sebagai: {class_labels[predicted_class]}")

```

## Project-folder/
│── animal-dataset/         # Dataset gambar hewan
│── dataset-final/          # Dataset final gambar hewan untuk model
│── saved_model/            # Model yang telah dikonversi ke SavedModel
│── tfjs_model/             # Model dalam format TensorFlow.js
│── model_mobile.tflite     # Model dalam format TF-Lite
│── requirements.txt        # Daftar library yang digunakan
│── README.md               # Dokumentasi proyek

