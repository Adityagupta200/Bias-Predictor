import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
import tensorflow as tf
import json
import pandas as py
import pandas as pd
from sklearn.metrics import accuracy_score
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from builtins import list
from sklearn.preprocessing import LabelEncoder

with open('/kaggle/input/stereoset/StereoSet.json', 'r') as f:
    d = json.load(f)

# Load the CSV file
file_path = '/kaggle/input/revised-ethos-dataset/Revised_Ethos_Dataset.csv'  # Replace with the actual file path
data_ethos = pd.read_csv(file_path)

# Open the file in read mode
with open("/kaggle/input/gender-race-religion-dataset/Gender_Bias_Sentences.txt", "r") as gender_bias_sentences:
    sentences_gender = gender_bias_sentences.read().splitlines()

with open("/kaggle/input/gender-race-religion-dataset/Race_Bias_Sentences.txt", "r") as race_bias_sentences:
    sentences_race = race_bias_sentences.read().splitlines()

with open("/kaggle/input/gender-race-religion-dataset/Religion_Bias_Sentences.txt", "r") as religion_bias_sentences:
    sentences_religion = religion_bias_sentences.read().splitlines()

labels = []
for i in range(0, 2123):
    b = d[i]
    a = b['bias_type']

    labels.append(a)
    labels.append(a)
    labels.append(a)

for i in range(0, 325):
    labels.append(data_ethos['dominant_bias'][i])

sentences_gender = list(set(sentences_gender))
sentences_race = list(set(sentences_race))
sentences_religion = list(set(sentences_religion))

for i in sentences_gender:
    labels.append('gender')
for i in sentences_race:
    labels.append('race')
for i in sentences_religion:
    labels.append('religion')

# print(np.array(labels).shape)

sentences = []
for i in range(0, 2123):

    are = d[i]
    are2 = are['sentences']
    for il in are2['sentence']:
        sentences.append(are['context'] + " " + il)
        i = i + 1

for i in range(0, 325):
    a = data_ethos['comment'][i]
    sentences.append(a)

for i in sentences_gender:
    sentences.append(i)
for i in sentences_race:
    sentences.append(i)
for i in sentences_religion:
    sentences.append(i)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)
joblib.dump(label_encoder, 'label_encoder.pkl')


# Prepare the text data and labels
texts = sentences  # The combined list of sentences
labels = labels # The corresponding labels

# Step 1: Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

joblib.dump(tokenizer, 'tokenizer.pkl')

# Step 2: Pad the sequences to have the same length
max_len = max([len(seq) for seq in sequences])  # Find the max length of the sentences
X = pad_sequences(sequences, maxlen=max_len)

# Step 4: Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Create an instance of SMOTE for oversampling
smote = SMOTE(sampling_strategy='auto', random_state=42)

# Apply SMOTE on the training data
X_train_oversampled, y_train_oversampled = smote.fit_resample(X_train, y_train)

# Convert oversampled labels back to one-hot encoding
y_train_oversampled = to_categorical(y_train_oversampled)
y_val = to_categorical(y_val)  # Ensure validation labels are also one-hot encoded

joblib.dump(y_val,'testing_labels.pkl')
joblib.dump(X_val,'testing_sentences.pkl')

# Step 5: Build the LSTM model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_len))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(64))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))  # Output units match number of classes

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_oversampled, y_train_oversampled, epochs=15, batch_size=32, validation_data=(X_val, y_val))

# Optionally, save the model
model.save('bias_predict_model.keras')