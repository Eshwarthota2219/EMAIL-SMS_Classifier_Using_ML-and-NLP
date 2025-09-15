import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# Initialize stemmer
ps = PorterStemmer()

# Text preprocessing
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [i for i in text if i.isalnum()]
    text = y[:]
    y.clear()

    y = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]
    text = y[:]
    y.clear()

    y = [ps.stem(i) for i in text]
    return " ".join(y)

# Load model and vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit App
st.set_page_config(page_title="Spam Classifier", page_icon="📧", layout="centered")

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📧 Email/SMS Spam Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter a message below to check if it is Spam or Ham</p>", unsafe_allow_html=True)

# Sidebar info
st.sidebar.title("About")
st.sidebar.info("""
This app classifies your messages as **Spam** or **Ham** using a machine learning model.
- Enter any email or SMS in the text box
- Click 'Predict'
- Get instant classification results
""")

# Input
input_sms = st.text_area("Enter your message:", height=150, placeholder="Type or paste your message here...")

if st.button("Predict"):
    if input_sms.strip() != "":
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        prob = model.predict_proba(vector_input)[0]

        # Display results with colored boxes
        if result == 1:
            st.markdown(f"<div style='padding:10px; background-color:#FFCDD2; color:#B71C1C; border-radius:5px;'><h3>This message is Spam ❌</h3></div>", unsafe_allow_html=True)
            st.write(f"Confidence: {prob[1]*100:.2f}%")
        else:
            st.markdown(f"<div style='padding:10px; background-color:#C8E6C9; color:#1B5E20; border-radius:5px;'><h3>This message is Ham ✅</h3></div>", unsafe_allow_html=True)
            st.write(f"Confidence: {prob[0]*100:.2f}%")

        # Optional: show transformed text
        with st.expander("Show Preprocessed Text"):
            st.write(transformed_sms)
    else:
        st.warning("Please enter a message to classify.")
