import joblib
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from collections import Counter
import re,string,unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import WordNetLemmatizer
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from bs4 import BeautifulSoup
from collections import Counter
from sklearn.model_selection import train_test_split
import re,string,unicodedata

import streamlit as st
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from textblob import TextBlob
from textblob import Word
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

# Tokenization of text
tokenizer = ToktokTokenizer()
# Setting English stopwords
stop_words = set(stopwords.words('english'))

# Lemmatization of text
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # Convert to string and lowercase
    text = str(text).lower()

    # Remove the html strips
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    
    # Removing square brackets
    text = re.sub('\[[^]]*\]', '', text)

    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Remove single character
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)

    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)

    # Lemmatizing the text
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())
    
    # Remove stopwords
    tokens = tokenizer.tokenize(text)

    tokens = [token.strip() for token in tokens]

    clean_tokens = [token for token in tokens if token not in stop_words]
    
    clean_text = ' '.join(clean_tokens)

    return clean_text

X_train_tv = joblib.load('analysis/X_train_tv.pkl')
X_test_tv = joblib.load('analysis/X_test_tv.pkl')
y_train_labelled = joblib.load('analysis/y_train_labelled.pkl')
y_test_labelled = joblib.load('analysis/y_test_labelled.pkl')

# def get_train_test_data():
#     df = pd.read_csv("analysis/IMDB Dataset.csv")
#     df['review'] = df['review'].apply(clean_text)
#     X = df['review']
#     y = df['sentiment']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     return X_train, X_test, y_train , y_test

# def vectorize_and_label_data():
#     # tv = TfidfVectorizer(use_idf=True, ngram_range=(1,3))
#     # X_train, X_test, y_train , y_test = get_train_test_data()

#     X_train_tv = tv.fit_transform(X_train)
#     X_test_tv = tv.transform(X_test)

#     lb = LabelBinarizer()
#     y_train_labelled = lb.fit_transform(y_train)
#     y_test_labelled = lb.transform(y_test)
#     return X_train_tv, X_test_tv,y_train_labelled, y_test_labelled

# Logistic Regression 

def get_lr_model():
    lr = LogisticRegression(penalty='l2', max_iter=1000,C=1.0, random_state=42)

    # X_train_tv, X_test_tv,y_train_labelled, y_test_labelled = vectorize_and_label_data()
    lr_tfidf = lr.fit(X_train_tv, y_train_labelled)

    lr_predict = lr_tfidf.predict(X_test_tv)

    lr_accuracy = accuracy_score(y_test_labelled, lr_predict)

    cm_tfidf_lr = confusion_matrix(y_test_labelled, lr_predict, labels=[1,0])

    return lr_tfidf, lr_accuracy, cm_tfidf_lr

# SGD Classifier

def get_sgd_model():
    svm = SGDClassifier(loss='hinge', max_iter=1000, random_state=42)
    # X_train_tv, X_test_tv,y_train_labelled, y_test_labelled = vectorize_and_label_data()
    svm_tfidf = svm.fit(X_train_tv, y_train_labelled)

    svm_predict = svm_tfidf.predict(X_test_tv)

    svm_accuracy = accuracy_score(y_test_labelled, svm_predict)

    cm_tfidf_svm = confusion_matrix(y_test_labelled, svm_predict, labels=[1,0])

    return svm_tfidf, svm_accuracy, cm_tfidf_svm

# Multinomial Naive Bayes

def get_nb_model():
    mnb = MultinomialNB()
    # X_train_tv, X_test_tv,y_train_labelled, y_test_labelled = vectorize_and_label_data()
    mnb_tfidf = mnb.fit(X_train_tv, y_train_labelled)

    
    mnb_predict = mnb.predict(X_test_tv)

    mnb_accuracy = accuracy_score(y_test_labelled, mnb_predict)

    cm_tfidf_mnb = confusion_matrix(y_test_labelled, mnb_predict, labels=[1,0])

    return mnb_tfidf, mnb_accuracy, cm_tfidf_mnb


# Create a gauge chart to display the accuracy scores of models
def create_gauge_chart(model_name, accuracy):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=accuracy*100,
        title={"text": model_name},
        gauge={
            "axis": {"range": [None, 100]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, 50], "color": "darkslategray"},
                {"range": [50, 75], "color": "darkgray"},
                {"range": [75, 100], "color": "lightgray"}
            ],
        }
    ))
    return fig


# Function to plot confusion matrix of the chosen model

def plot_confusion_matrix(cm, model_name):
    fig = px.imshow(cm,
                    text_auto=True,
                    labels={'x': 'Model Prediction', 'y': 'Actual Sentiment'},
                    x=['Positive Sentiment', 'Negative Sentiment'],  # Positive is first
                    y=['Positive Sentiment', 'Negative Sentiment'],
                    color_continuous_scale='Blues',
                    title=f'Confusion Matrix for {model_name}'
                    )
    
    return fig