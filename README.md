# Bias Prediction App

## Project Description
This project provides a web application that predicts the type of bias in a given sentence. It leverages a pre-trained machine learning model built with TensorFlow and Keras to classify user input into different bias categories. The app is designed for potential integration into applications like chatting platforms or social media services to monitor and flag biased or offensive content. The application also uses the `transformers` library to detect offensive language.

## Description of Datasets

### 1. **`StereoSet.json`**
   - **Description**: A dataset that evaluates biases in natural language models related to stereotypes in various domains, such as gender, race, religion, and profession.
   - **Usage**: Used for training and evaluating the model on stereotype detection and classification tasks.

### 2. **`Gender_Bias_Sentences.txt`**
   - **Description**: Contains sentences showcasing explicit and implicit gender biases, such as stereotyping or unfair assumptions based on gender.
   - **Usage**: Provides data to train the model to identify gender-related biases in input text.

### 3. **`Race_Bias_Sentences.txt`**
   - **Description**: Contains sentences reflecting racial biases, including discriminatory remarks or prejudiced assumptions based on ethnicity.
   - **Usage**: Used to train the model to detect racial bias in text.

### 4. **`Religion_Bias_Sentences.txt`**
   - **Description**: Includes sentences with religious biases, such as judgments, stereotypes, or hostility toward specific religions.
   - **Usage**: Helps train the model to identify religious bias in input text.

### 5. **`Revised_Ethos_Dataset.csv`**
   - **Description**: A comprehensive dataset combining various bias categories, including toxicity, hate speech, and stereotypes. Revised for better accuracy and usability in training.
   - **Usage**: Used as a supplementary dataset to enhance model performance and generalization.

## Description of Files and Directories

- **`app.py`**: The main Python script containing logic for loading the model, tokenizer, and label encoder. It manages the Flask app routes, processes user input, and displays predictions.

- **`Procfile`**: Specifies the command used to start the application on platforms like Render or Heroku. Contains `web: python app.py`.

- **`requirements.txt`**: Lists all the Python dependencies required for the project. These can be installed using `pip install -r requirements.txt`.

- **`model/`**: Contains the following pre-trained artifacts:
  - `bias_predict_model.keras`: The machine learning model for bias prediction.
  - `tokenizer.pkl`: Tokenizer object for preprocessing text inputs.
  - `label_encoder.pkl`: Encodes and decodes the output classes for bias types.

- **`templates/`**: Contains the `index.html` file, which provides the web interface for users to input text and view predictions.

- **`.idea/`**: PyCharm-specific configuration files. These are not necessary for deployment and can be excluded from the repository.

## Implementation in Applications

### **Chat Applications**
- Use the bias prediction model to process user messages in real-time. 
  1. Convert the message entered by the user into a string.
  2. Add the string to a list (e.g., `[message]`) and preprocess it using the tokenizer provided in the model code.
  3. Use the trained model to predict the bias type and respond accordingly, such as by flagging the message for moderation or giving immediate feedback to the user.

### **Social Media Platforms**
- Integrate the model into backend systems to automatically analyze posts and comments before publishing.
  1. Pass user-generated content through the model by preprocessing it as outlined in the shared code.
  2. If the content is detected as biased or offensive, prevent it from being posted or display a warning to the user about community guidelines.

### **Customer Support Chatbots**
- Enhance chatbot interactions by filtering potentially biased or offensive user inputs.
  1. Preprocess each input from the user and predict its bias type.
  2. If bias is detected, block the message and ask the user to rephrase, or anonymize the input before it reaches a support agent.
  3. For chatbot-generated responses, validate the response to ensure it remains neutral and unbiased.

### **Educational Tools**
- Utilize the bias detection model in tools aimed at raising awareness about bias in language.
  1. Allow users to input sentences and observe real-time feedback from the model.
  2. Display explanations for why a sentence may be considered biased to encourage learning and awareness.

## Setup Instructions

To set up and run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone <repository-url>
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

   You can create a `.env` file to manage these variables more easily.

4. Run the Flask app:
   ```bash
   python app.py
   ```

5. Open your web browser and navigate to `http://127.0.0.1:5000/` to use the app.

## Example Usage

1. Enter a sentence in the input box on the web interface (e.g., "Why is he working as a nurse? That’s a woman’s job.").
2. Click the "Predict Bias" button.
3. The app processes the input and displays the predicted bias type (e.g., "Gender Bias") or indicates "None" if no bias is detected.

## Video Demonstration

Here is a video demonstrating how the Bias Prediction App works:

[Video Link Placeholder]  
(Insert link to your project demo video here)

