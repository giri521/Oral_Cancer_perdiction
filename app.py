from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import uuid
import gdown
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Model path and download from Google Drive if missing
model_path = 'oral_cancer_detection_model.h5'
google_drive_file_id = '1EhG7tzuXJzPHxTbneFs8PTINjuvxaexx'

if not os.path.exists(model_path):
    print("[INFO] Downloading model from Google Drive...")
    url = f'https://drive.google.com/uc?id={google_drive_file_id}'
    gdown.download(url, model_path, quiet=False)
    print("[INFO] Model downloaded successfully.")

# Load the trained model
model = tf.keras.models.load_model(model_path)
print("[INFO] Model loaded.")

# Ensure 'static' folder exists to save uploaded images
if not os.path.exists('static'):
    os.makedirs('static')

# Prediction function
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        return "No Oral Cancer", float(prediction)
    else:
        return "Oral Cancer Detected", float(prediction)

# Home route
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files.get("file")
        if file:
            filename = secure_filename(file.filename)
            unique_name = f"{uuid.uuid4().hex}_{filename}"
            file_path = os.path.join("static", unique_name)
            file.save(file_path)

            result, confidence = predict_image(file_path)

            return render_template("result.html",
                                   prediction=result,
                                   confidence=round(confidence * 100, 2),
                                   image_path=file_path)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
