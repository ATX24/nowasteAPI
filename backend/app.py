from flask import Flask, request
import cv2
import tensorflow as tf
import numpy as np
from flask_cors import CORS



app = Flask(__name__)
CORS(app)

# Load the model
model = tf.keras.models.load_model("model.h5")

@app.route("/predict", methods=["POST"])
def predict():
    # Get the image from the request
    image = request.files["image"].read()

    # Convert the image to a numpy array
    image = cv2.imdecode(np.fromstring(image, np.uint8), cv2.IMREAD_UNCHANGED)

    # Preprocess the image
    image = cv2.resize(image, (227, 227))
    image = image / 255.0

    # Make predictions using the model
    predictions = model.predict(image[np.newaxis, ...])

    # Get the class with the highest probability
    class_idx = np.argmax(predictions[0])

    # Return the result
    if class_idx == 0:
        return "Not recyclable"
    else:
        return "Recyclable"

if __name__ == "__main__":
    app.run()
