import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import streamlit as st

# Data Cleaning and Preprocessing
def preprocess(text):
    return text.lower()

data = pd.read_csv("spam.csv", encoding= 'latin-1')
data = data[["class", "message"]]
data['message'] = data['message'].apply(preprocess)

x = np.array(data["message"])
y = np.array(data["class"])

# Using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Training the model
clf = MultinomialNB(alpha=0.5)
clf.fit(X_train, y_train)

# Model Evaluation
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label="spam")
recall = recall_score(y_test, y_pred, pos_label="spam")
f1 = f1_score(y_test, y_pred, pos_label="spam")

# Streamlit UI
st.title("Improved Spam Detection System")
st.write(f"Model Accuracy: {accuracy:.2f}")
st.write(f"Model Precision: {precision:.2f}")
st.write(f"Model Recall: {recall:.2f}")
st.write(f"Model F1 Score: {f1:.2f}")

def spamdetection():
    user = st.text_area("Enter any Message or Email: ")
    if len(user) < 1:
        st.write("  ")
    else:
        sample = preprocess(user)
        data = vectorizer.transform([sample])
        prediction = clf.predict(data)
        prob = clf.predict_proba(data)
        st.write(f"Probability of being Spam: {prob[0][1]:.2f}")
        st.write(f"Prediction: {'Spam' if prediction[0] == 'spam' else 'Not Spam'}")

spamdetection()





