import streamlit as st
import joblib
import h5py
import numpy as np
import os
from PIL import Image

# File paths
vectorizer_filename = 'artifacts/data_transformation/count_vectorizer.pkl'
model_filename = 'artifacts/model_trainer/model.h5'

# Function to load vectorizer and model
def load_model_and_vectorizer():
    if os.path.exists(vectorizer_filename) and os.path.exists(model_filename):
        loaded_vectorizer = joblib.load(vectorizer_filename)
        loaded_model = h5py.File(model_filename, 'r')
        return loaded_vectorizer, loaded_model
    else:
        return None, None

# Function to predict email content
def predict_email_content(email_content, vectorizer, model):
    email_vectorized = vectorizer.transform([email_content])
    email_vectorized = email_vectorized.toarray()
    prediction = model['model_weights']['dense_1']['dense_1']['kernel:0'][:].dot(email_vectorized.T) + model['model_weights']['dense_1']['dense_1']['bias:0'][:]
    prediction = 1 / (1 + np.exp(-prediction))  # Sigmoid activation
    predicted_class = (prediction > 0.5).astype(int)
    return predicted_class[0][0]

# Load vectorizer and model
loaded_vectorizer, loaded_model = load_model_and_vectorizer()

# Streamlit App
st.title("Spam Email Classifier")

st.markdown("""
## Project Overview
This project uses deep learning to classify emails as spam or ham. It involves data preparation, text vectorization using CountVectorizer, and building a neural network model. The model achieves 98% accuracy using techniques like dense layers, dropout, and evaluation metrics such as precision and recall.

## Importance of Spam Classifiers
- **Security**: Prevents phishing and malicious attacks.
- **Productivity**: Focus on important emails.
- **Resource Management**: Saves server resources and reduces storage costs.
- **User Experience**: Keeps inboxes clean and organized.
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

st.markdown("---")
st.markdown("### Resources regarding the above project can be found below:")

col1, col2, col3 = st.columns(3)

with col1:
    github_icon = Image.open('assests\github-mark.png')
    st.image(github_icon, width=50)
    st.markdown("[GitHub](https://github.com/Harshithvarma007/Spam-email)", unsafe_allow_html=True)

with col2:
    kaggle_icon = Image.open('assests/4373210_kaggle_logo_logos_icon.png')
    st.image(kaggle_icon, width=50)
    st.markdown("[Kaggle](https://www.kaggle.com/code/harshithvarma007/spam-email-classification-98-accuracy)", unsafe_allow_html=True)

with col3:
    medium_icon = Image.open('assests\medium.png')
    st.image(medium_icon, width=50)
    st.markdown("[Medium](https://medium.com/@harshith007varma007/end-to-end-machine-learning-project-part-i-c29c2b982055)", unsafe_allow_html=True)

# To run the Streamlit app, save this script as `app.py` and run the following command:
# streamlit run app.py
