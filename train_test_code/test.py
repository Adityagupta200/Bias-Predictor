from sklearn.metrics import accuracy_score
import joblib
import tensorflow as tf
from sklearn.metrics import f1_score

model= tf.keras.models.load_model('/kaggle/working/bias_predict_model.keras')

X_val = joblib.load('/kaggle/working/testing_sentences.pkl')
y_val = joblib.load('/kaggle/working/testing_labels.pkl')

# Evaluate the model on the validation set
y_val_pred_probs = model.predict(X_val)
y_val_pred = y_val_pred_probs.argmax(axis=1)  # Convert probabilities to class labels
y_val_true = y_val.argmax(axis=1)  # Convert one-hot encoded true labels to class labels

# Calculate the Macro-F1 score
macro_f1 = f1_score(y_val_true, y_val_pred, average='macro')
print(f"Macro F1 Score: {macro_f1:.4f}")

# Calculate and print accuracy
test_accuracy = accuracy_score(y_val_true, y_val_pred)
print(f"Test Accuracy: {test_accuracy:.2f}")