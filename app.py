# app.py
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the saved model
model = tf.keras.models.load_model('oral_cancer_detection_model.h5')

# Create static directory if it doesn't exist
if not os.path.exists('static'):
    os.makedirs('static')

# Prediction function
def predict_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(128, 128))  # Resize image to 128x128
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image
    
    # Get model prediction
    prediction = model.predict(img_array)
    
    # If the prediction is greater than 0.5, it's "No Oral Cancer", otherwise "Oral Cancer Detected"
    if prediction[0] > 0.5:
        return "No Oral Cancer"  # Change this to what your model should output for "No Cancer"
    else:
        return "Oral Cancer Detected"  # This will be for cases where cancer is detected

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Check if file is uploaded
        file = request.files["file"]
        if file:
            # Save the uploaded file to the 'static' folder
            file_path = f"static/{file.filename}"
            file.save(file_path)

            # Get the prediction result
            result = predict_image(file_path)

            # Render the result page with the prediction and image
            return render_template("result.html", prediction=result, image_path=file_path)

    # Render the main upload page
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
