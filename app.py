import streamlit as st
import string
import pickle

from pycparser import preprocess_file

tfidf = pickle.load(open('tfidf2.pkl','rb'))
model = pickle.load(open('Spam_Emails_classifiers.pkl','rb'))

st.title("Spam Emails Classifier")
input_emails = st.text_input("Enter The Message")

if st.button('Predict'):
    # 1. preprocess_file -->
    input_emails = input_emails.lower()
    # 2.  vectorize -->
    tfidf_input = tfidf.transform([input_emails])
    # 3.  Predict -->
    result = model.predict(tfidf_input)[0]
    # 4. Display -->
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spame")

