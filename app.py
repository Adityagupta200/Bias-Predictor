# Copyright (c) [2025] [ADITYA GUPTA]
# Licensed under the MIT License. See LICENSE for details.

import os
import joblib
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load paths from environment variables
model_path = os.getenv("MODEL_PATH", "model/bias_predict_model.keras")
tokenizer_path = os.getenv("TOKENIZER_PATH", "model/tokenizer.pkl")
label_encoder_path = os.getenv("LABEL_ENCODER_PATH", "model/label_encoder.pkl")

# Load your trained model, tokenizer, and label encoder
model = tf.keras.models.load_model(model_path)
tokenizer = joblib.load(tokenizer_path)
label_encoder = joblib.load(label_encoder_path)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form["text_input"]
        if user_input.strip() == "":
            return render_template("index.html", result="Please enter some text!")

        input_vector_sequences = tokenizer.texts_to_sequences([user_input])
        max_len = max([len(seq) for seq in input_vector_sequences])
        X = pad_sequences(input_vector_sequences, maxlen=max_len)

        predictions = model.predict(X)
        predicted_classes = np.argmax(predictions, axis=1)
        predicted_labels = label_encoder.inverse_transform(predicted_classes)

        result = f"Predicted Bias Type: {predicted_labels[0]}"
        return render_template("index.html", result=result)

    return render_template("index.html", result="")

if __name__ == "__main__":
    app.run(debug=True)
