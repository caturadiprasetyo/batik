from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Daftar kelas batik
classes = ['Kawung', 'Mega Mendung', 'Parang', 'Truntum']

# Load model
try:
    model = load_model('my_model.h5')
except Exception as e:
    print("Error loading model:", e)
    model = None

def predict_label(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)
        class_index = np.argmax(prediction)

        return classes[class_index]
    except Exception as e:
        print("Error during prediction:", e)
        return None

# Halaman utama
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("classification.html")

# Halaman untuk upload gambar dan mendapatkan prediksi
@app.route("/submit", methods=['POST'])
def get_output():
    if request.method == 'POST' and 'my_image' in request.files:
        img = request.files['my_image']
        
        if img.filename == '':
            return render_template("classification.html", prediction="No file selected", img_path=None)
        
        filename = secure_filename(img.filename)
        img_path = os.path.join("static", filename)
        img.save(img_path)

        prediction = predict_label(img_path)
        
        return render_template("classification.html", prediction=prediction, img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)
