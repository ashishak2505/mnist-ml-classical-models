from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("../models/mnist_sgd_model.joblib")

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
