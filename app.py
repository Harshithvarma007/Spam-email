import streamlit as st
import joblib
import tensorflow as tf
import os
from PIL import Image

# File paths
vectorizer_filename = 'harshithvarma007/spam-email/main/artifacts/data_transformation/count_vectorizer.pkl'
model_filename = 'harshithvarma007/spam-email/main/artifacts/model_trainer/model.h5'

# Function to load vectorizer and model
def load_model_and_vectorizer():
    if os.path.exists(vectorizer_filename) and os.path.exists(model_filename):
        loaded_vectorizer = joblib.load(vectorizer_filename)
        loaded_model = tf.keras.models.load_model(model_filename)
        return loaded_vectorizer, loaded_model
    else:
        return None, None

# Function to predict email content
def predict_email_content(email_content, vectorizer, model):
    email_vectorized = vectorizer.transform([email_content])
    prediction = model.predict(email_vectorized)
    predicted_class = (prediction > 0.5).astype(int)
    return predicted_class[0][0]

# Load vectorizer and model
loaded_vectorizer, loaded_model = load_model_and_vectorizer()

# Streamlit App
st.title("Spam Email Classifier")

st.markdown("""
## Spam Email Classification using Deep Learning

### Project Overview
This project focuses on developing a spam email classifier using a deep learning approach. The goal is to create a model that can effectively distinguish between spam (unwanted emails) and ham (legitimate emails). The process involves multiple steps, from data preparation and cleaning to building and evaluating a neural network model.

### Importance of Spam Classifiers
Spam email classifiers are crucial in today's digital age due to the following reasons:
- **Security**: Spam emails often contain phishing links or malicious attachments that can compromise security and lead to data breaches.
- **Productivity**: Filtering out spam helps users focus on important emails, thereby improving productivity.
- **Resource Management**: Effective spam filtering saves server resources and reduces storage costs by preventing the accumulation of unwanted emails.
- **User Experience**: A good spam filter enhances the user experience by keeping the inbox clean and organized.

### Techniques Used
To achieve a high accuracy of 98%, the project employs several sophisticated techniques:

**Data Cleaning**:
- **Removing Duplicates**: Duplicate emails are removed to ensure data quality and avoid bias in the model.
- **Handling Missing Values**: Although no missing values were found in this dataset, checking for them is a crucial step to ensure the integrity of the data.

**Text Vectorization**:
- **CountVectorizer**: This technique transforms the text data into numerical features by counting the frequency of words in each email. This is essential for feeding the data into a neural network, as machine learning models require numerical input.

**Building a Deep Learning Model**:
- **Neural Network Architecture**: The model consists of several dense layers with ReLU activation functions and a dropout layer to prevent overfitting. The architecture is designed to capture the complex patterns in the email text data.
- **Dense Layers**: These layers are fully connected, meaning each neuron receives input from all neurons of the previous layer, enabling the model to learn intricate relationships in the data.
- **Dropout Layer**: This layer randomly sets a fraction of input units to 0 at each update during training time, which helps prevent overfitting by making the model more robust.

**Model Training and Evaluation**:
- **Training**: The model is trained on the vectorized text data using the Adam optimizer and binary cross-entropy loss function. Training involves multiple epochs to adjust the weights of the neural network to minimize the loss.
- **Evaluation Metrics**:
    - **Accuracy**: Measures the proportion of correctly classified emails.
    - **Precision**: Indicates the proportion of true positive identifications among the predicted positives, which is critical for spam detection to minimize false positives.
    - **Recall**: Reflects the proportion of true positive identifications among all actual positives, ensuring that most spam emails are detected.
    - **F1 Score**: A harmonic mean of precision and recall, providing a single metric to balance the two.

**Confusion Matrix**:
- **Visualization**: The confusion matrix provides a detailed breakdown of the model's performance, showing the counts of true positives, true negatives, false positives, and false negatives. This helps in understanding the areas where the model performs well and where it needs improvement.

**Saving and Reusing the Model**:
- **Joblib and TensorFlow**: The trained model and the vectorizer are saved to disk using joblib and TensorFlow's save functionality. This ensures that the model can be reused for predictions without retraining, saving time and computational resources.

By following these techniques and methodologies, the project successfully achieves a high accuracy rate, demonstrating the effectiveness of deep learning in spam email classification.
""")

# Text box for email content
email_content = st.text_area("Enter the email content:", "")

# Predict button
if st.button("Predict"):
    if loaded_vectorizer and loaded_model:
        if email_content:
            prediction = predict_email_content(email_content, loaded_vectorizer, loaded_model)
            if prediction == 1:
                st.markdown("<span style='color:red'>Spam</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span style='color:green'>Ham</span>", unsafe_allow_html=True)
        else:
            st.write("Please enter the email content.")
    else:
        st.write("Model and vectorizer not found. Please train the model first.")
# Add GitHub, Kaggle, and Medium icons with links
st.markdown("---")
st.markdown("### Resources regarding the above project can be found below:")

col1, col2, col3 = st.columns(3)

with col1:
    github_icon = Image.open('harshithvarma007/spam-email/main/assests\github-mark.png')
    st.image(github_icon, width=50)
    st.markdown("[GitHub](https://github.com/Harshithvarma007/Spam-email)", unsafe_allow_html=True)

with col2:
    kaggle_icon = Image.open('harshithvarma007/spam-email/main/assests/4373210_kaggle_logo_logos_icon.png')
    st.image(kaggle_icon, width=50)
    st.markdown("[Kaggle](https://www.kaggle.com/code/harshithvarma007/spam-email-classification-98-accuracy)", unsafe_allow_html=True)

with col3:
    medium_icon = Image.open('harshithvarma007/spam-email/main/assests\medium.png')
    st.image(medium_icon, width=50)
    st.markdown("[Medium](https://medium.com/@harshith007varma007/end-to-end-machine-learning-project-part-i-c29c2b982055)", unsafe_allow_html=True)

# To run the Streamlit app, save this script as `app.py` and run the following command:
# streamlit run app.py
