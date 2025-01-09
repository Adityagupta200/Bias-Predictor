from flask import Flask, request, render_template
import joblib
import numpy as np
import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load your trained model and tokenizer
model = tf.keras.models.load_model(r"C:\Users\abc\Downloads\bias_predict_model.keras")  # Path to your model
tokenizer = joblib.load(r"C:\Users\abc\Downloads\tokenizer.pkl")  # Path to your TF-IDF tokenizer
label_encoder = joblib.load(r'C:\Users\abc\Downloads\label_encoder.pkl')

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the user input from the form
        user_input = request.form["text_input"]

        if user_input.strip() == "":
            return render_template("index.html", result="Please enter some text!")

        # Convert text to sequences
        input_vector_sequences = tokenizer.texts_to_sequences([user_input])

        max_len = max([len(seq) for seq in input_vector_sequences])  # Find the max length of the sentences
        X = pad_sequences(input_vector_sequences, maxlen=max_len)

        # Predict the bias type
        predictions = model.predict(X)
        predicted_classes = np.argmax(predictions, axis=1)

        # Step 4: Map the predicted class indices back to the original bias types
        predicted_labels = label_encoder.inverse_transform(predicted_classes)

        # Return the result to the user
        result = f"Predicted Bias Type: {predicted_labels[0]}"
        return render_template("index.html", result=result)

    # Render the default page
    return render_template("index.html", result="")

if __name__ == "__main__":
    app.run(debug=True)
