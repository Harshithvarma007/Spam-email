import streamlit as st
import joblib
import tensorflow as tf
import subprocess
import os

# File paths
vectorizer_filename = 'artifacts/data_transformation/count_vectorizer.pkl'
model_filename = 'artifacts/model_trainer/model.h5'

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

# Train model button
if st.button("Train Model"):
    with st.spinner('Training the model...'):
        subprocess.run(["python", "main.py"])
    st.success("Model training completed.")
    # Reload vectorizer and model after training
    loaded_vectorizer, loaded_model = load_model_and_vectorizer()

# To run the Streamlit app, save this script as `app.py` and run the following command:
# streamlit run app.py
