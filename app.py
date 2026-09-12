import streamlit as st
import pickle
import re

def preprocess_message(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

st.title("Spam Message Detection System")

message = st.text_area("Enter a message to classify:")

with open("spam_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

if st.button("Check Message"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        clean_message = preprocess_message(message)
        message_tfidf = vectorizer.transform([clean_message])
        prediction = model.predict(message_tfidf)[0]

        if prediction == 1:
            st.error("This message is SPAM.")
        else:
            st.success("This message is HAM (not spam).")