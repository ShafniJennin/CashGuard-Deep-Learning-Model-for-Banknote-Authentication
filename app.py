# import os
# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# from flask import Flask, request, render_template, jsonify
# from werkzeug.utils import secure_filename

# app = Flask(__name__)

# # Load the trained model
# model = load_model(r"C:\Users\Shafni\Documents\Shafni-PG-PROJECT\FakeCurrencyDetection\currency_classifier.h5")

# # Upload folder configuration
# UPLOAD_FOLDER = os.path.join("static", "uploads")
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# # Allowed image extensions
# ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# def allowed_file(filename):
#     return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# def preprocess_image(image_path):
#     """ Preprocess the image for model prediction """
#     img = cv2.imread(image_path)
#     if img is None:
#         return None  # Handle cases where the file is not an image
    
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
#     img = cv2.resize(img, (128, 128))  # Resize to match model input (128x128)
#     img = img.astype("float32") / 255.0  # Normalize
#     img = np.expand_dims(img, axis=0)  # Add batch dimension
#     return img

# @app.route("/")
# def index():
#     return render_template("index.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files["file"]
#     if file.filename == "":
#         return jsonify({"error": "No selected file"}), 400
    
#     if not allowed_file(file.filename):
#         return jsonify({"error": "Invalid file type. Only PNG, JPG, and JPEG are allowed."}), 400

#     filename = secure_filename(file.filename)  # Secure the filename
#     file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
#     file.save(file_path)  # Save uploaded image

#     img = preprocess_image(file_path)
#     if img is None:
#         return jsonify({"error": "Invalid image file. Please upload a valid image."}), 400

#     prediction = model.predict(img)
#     confidence = float(prediction[0][0])  # Extract confidence score

#     # Assuming Sigmoid activation: If confidence > 0.5, it's "Real"; else "Fake"
#     result = f"Real ({confidence:.2%} confidence)" if confidence > 0.4 else f"Fake ({(1 - confidence):.2%} confidence)"

#     return jsonify({"prediction": result})

# if __name__ == "__main__":
#     app.run(debug=True)

import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ✅ Load trained model
model = load_model(r"C:\Users\Shafni\Documents\Shafni-PG-PROJECT\FakeCurrencyDetection\currency_classifier.h5")

# ✅ Upload folder
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ✅ Allowed extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    """Preprocess image for prediction"""
    img = cv2.imread(image_path)

    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    return img


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ✅ Check file
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        if not allowed_file(file.filename):
            return jsonify({
                "error": "Invalid file type. Only PNG, JPG, JPEG allowed."
            }), 400

        # ✅ Save file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # ✅ Preprocess
        img = preprocess_image(file_path)

        if img is None:
            return jsonify({
                "error": "Invalid image. Could not process."
            }), 400

        # ✅ Predict
        prediction = model.predict(img)[0][0]  # sigmoid output

        # ✅ Decision logic
        if prediction > 0.5:
            label = "Genuine"
            confidence = prediction
        else:
            label = "Fake"
            confidence = 1 - prediction

        confidence_percent = round(confidence * 100, 2)

        # ✅ Return clean JSON
        return jsonify({
            "prediction": label,
            "confidence": confidence_percent
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({
            "error": "Something went wrong during prediction"
        }), 500


if __name__ == "__main__":
    app.run(debug=True)