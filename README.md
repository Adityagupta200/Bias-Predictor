# Bias Prediction App

## Project Description
This project provides a web application that predicts the type of bias in a given sentence. It uses a trained machine learning model built with TensorFlow and Keras to classify user input into different bias categories. The app is built using Flask for serving the web interface and machine learning predictions.

## Description of Files and Directories

- **`app.py`**: The main Python file containing the logic for loading the model, tokenizer, and label encoder. It handles the routing for the Flask app, taking user input, predicting bias types, and rendering the result back on the web page.

- **`Procfile`**: This file is used by platforms like Render or Heroku to run the application. It tells the platform how to start the web service. In this case, it contains `web: python app.py`.

- **`requirements.txt`**: Contains the list of Python dependencies needed to run the app. This file is used to install all the necessary libraries using `pip install -r requirements.txt`.

- **`model/`**: A directory that contains the following files:
  - `bias_predict_model.keras`: The pre-trained machine learning model file.
  - `tokenizer.pkl`: A pickled object containing the tokenizer used to preprocess input text.
  - `label_encoder.pkl`: A pickled object that encodes and decodes prediction classes.

- **`templates/`**: Contains the `index.html` file which provides the front-end interface for users to interact with the app. The form takes the user input and displays the predicted bias type.

- **`.idea/`**: This directory contains PyCharm-specific project files. It is generally not necessary for deployment and can be ignored when pushing to a public repository.

## Setup Instructions

To set up and run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone <[[repository-url]([https://github.com/Adityagupta200/Bias-Predictor](https://github.com/Adityagupta200/Bias-Predictor/tree/master))](https://github.com/Adityagupta200/Bias-Predictor/tree/master)>
   cd <project-directory>
   ```

2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - For local development, set the following environment variables:
     - `MODEL_PATH`: Path to the saved model (e.g., `model/bias_predict_model.keras`)
     - `TOKENIZER_PATH`: Path to the tokenizer (e.g., `model/tokenizer.pkl`)
     - `LABEL_ENCODER_PATH`: Path to the label encoder (e.g., `model/label_encoder.pkl`)

   You can set these in your local terminal session, or create a `.env` file for easier management.

4. Run the Flask app:
   ```bash
   python app.py
   ```

5. Open your web browser and navigate to `http://127.0.0.1:5000/` to use the app.

## Example Usage

- Enter a sentence in the input box, and click the "Predict Bias" button.
- The app will process the input and display the predicted bias type on the same page.

## Video Demonstration

Here is a video demonstrating how the Bias Prediction App works:

[Video Link Placeholder]  
(Insert link to your project demo video here)
