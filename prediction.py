import streamlit as st
import joblib
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer

# Page configuration
st.set_page_config(
    page_title="Spam Detection",
    page_icon="assets/icon.png",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Function to load vectorizer and model
@st.cache(allow_output_mutation=True)
def load_vectorizer_and_model(vectorizer_path, model_path):
    vectorizer = joblib.load(vectorizer_path)
    model = tf.keras.models.load_model(model_path)
    return vectorizer, model

# Load vectorizer and model
vectorizer_path = 'assets/count_vectorizer.pkl'
model_path = 'assets/model.h5'
vectorizer, model = load_vectorizer_and_model(vectorizer_path, model_path)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Spam Detection"])

if page == "Home":
    st.title("Welcome to Spam Detection")
    st.write("This application helps in detecting spam emails using a pre-trained model.")
    st.write("Navigate to the Spam Detection page to start.")
    st.image("assets/icon.png", use_column_width=True)

elif page == "Spam Detection":
    st.title("Spam Detection")
    st.write("Enter your email content below to check if it is spam or not.")

    # Input for new emails
    new_emails = st.text_area("Enter email content here (one email per line)", height=200)

    if st.button("Predict"):
        # Split input into individual emails
        email_list = new_emails.split("\n")
        # Vectorize new emails
        new_emails_vectorized = vectorizer.transform(email_list)
        # Predict using the loaded model
        predictions = model.predict(new_emails_vectorized)
        predicted_classes = (predictions > 0.5).astype(int)

        st.write("### Predictions:")
        for email, prediction in zip(email_list, predicted_classes):
            st.write(f"**Email:** {email}\n**Prediction:** {'Spam' if prediction else 'Ham'}\n")

# To run this app, save it as streamlit_app.py and run the command below:
# streamlit run streamlit_app.py
