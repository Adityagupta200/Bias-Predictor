# Copyright (c) [2025] [ADITYA GUPTA]
# Licensed under the MIT License. See LICENSE for details.

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
from transformers import pipeline

# Example test data (you can replace this with your actual test data)
test_texts = [
    "Why is he working as a nurse? That’s a woman’s job.",
    "You look like Russian and speak like Indian. Both are disgusting go kill yourself",
    "Hinduism is best religion",
    "People from that religion are always extremists.",
    "He doesn’t celebrate our holidays; he must be unfriendly.",
    "They don’t believe in our god, so they must be immoral.",
    "She wears a headscarf; she must be oppressed.",
    "He’s from a minority religion; he probably doesn’t fit in.",
    "Their religion is so outdated and rigid.",
    "People from that faith are always causing trouble.",
    "They can’t be trusted; they follow a strange belief system.",
    "He’s from a different religion; he probably won’t understand our customs.",
    "Why can’t they just follow the majority religion?",
    "why does that african american monkey hold gun like that?"
]

# Step 1: Tokenize and pad the test sentences
tokenizer = joblib.load('/kaggle/input/trained_models/other/default/1/tokenizer.pkl')

# Tokenize
sequences = tokenizer.texts_to_sequences(test_texts)
max_len = 50  # Ensure the length is the same as during training
X_test = pad_sequences(sequences, maxlen=max_len)

# Load the trained model
model = tf.keras.models.load_model('/kaggle/input/trained_models/other/default/1/bias_predict_model.keras')

# Step 2: Use the trained model to predict the bias type for each test sentence
predictions = model.predict(X_test)  # Returns an array of shape (num_samples, num_classes)

# Step 3: Decode the predictions (convert from one-hot encoding to class indices)
predicted_classes = np.argmax(predictions, axis=1)  # axis=1 because we want the index of the max value across the classes

label_encoder = joblib.load('/kaggle/input/trained_models/other/default/1/label_encoder.pkl')

# Step 4: Map the predicted class indices back to the original bias types
predicted_labels = label_encoder.inverse_transform(predicted_classes)

# Load the pre-trained offensive language detection model
classifier = pipeline('text-classification', model='unitary/toxic-bert')

# Print the predictions
for text, label in zip(test_texts, predicted_labels):
    if classifier(text)[0]['score'] < 0.3:
        print(f"Text: {text} \nPredicted Bias Type: None \n")
    else:
        print(f"Text: {text} \nPredicted Bias Type: {label} \n")
