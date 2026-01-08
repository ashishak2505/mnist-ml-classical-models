import os
import joblib
from flask import Flask, render_template, request
import numpy as np  

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "mnist_sgd_model.joblib")

model = joblib.load(MODEL_PATH)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    pixels = request.form.get("pixels")
    pixels = list(map(int, pixels.split(",")))
    arr = np.array(pixels).reshape(1, -1)
    pred = model.predict(arr)[0]
    return f"Predicted Digit: {pred}"

if __name__ == "__main__":
    app.run(debug=True)
