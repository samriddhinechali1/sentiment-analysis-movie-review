import pickle
import time
import streamlit as st
from nltk.corpus import stopwords
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from collections import Counter
import re,string,unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import WordNetLemmatizer
from helper import *



def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    return plt

# Get all the models and vectorizer

vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1,3))

# with open('models/logistic_regressor.pkl', 'rb') as f:
#     lr_model = pickle.load(f)

# with open('models/sgd_classifier.pkl', 'rb') as f:
#     sgd_model = pickle.load(f)

# with open('models/naive_bayes.pkl', 'rb') as f:
#     nb_model = pickle.load(f)

# with open('models/accuracy_scores.pkl', "rb") as f:
#     accuracy_scores = pickle.load(f)

# with open('models/confusion_matrices.pkl', 'rb') as f:
#     confusion_matrices = pickle.load(f)

# Get all models , their accuracy scores and their confusion matrix

lr_model, logreg_accuracy, logreg_cm = get_lr_model()
sgd_model, sgd_accuracy, sgd_cm = get_sgd_model()
nb_model, nb_accuracy, nb_cm = get_nb_model()

st.set_page_config(page_title="Sentiment Analysis", page_icon="🎬", initial_sidebar_state="expanded")

# Set up the theme toggle
# Set up the theme toggle
theme_choice = st.sidebar.selectbox("Choose Theme", ["Dark", "Light"])

# Set the page config based on the theme selected
if theme_choice == "Dark":
    
    # Apply custom CSS for dark theme
    st.markdown("""
         <style>
                .stApp{
                    background-color: #1e1e1e;
                    color: white;
                }
                .sidebar .sidebar-content {
                    background-color: #333;
                }
                .sidebar .sidebar-content .stButton>button {
                    background-color: #555;
                    color: white;
                }
                .stTextArea>textarea {
                    background-color: #333;
                    color: white;
                    border: 1px solid #555;
                }
        </style>
    """, unsafe_allow_html=True)
else:
    
    # Apply custom CSS for light theme
    st.markdown("""
        <style>
            .stApp {
                background-color: white;
                color: black;
            }
            
            .sidebar .sidebar-content  {
                background-color: #ffffff;  /* Make sure sidebar content is white in light mode */
                color: black;
                border: 1px solid #ccc;
            }
            .stButton>button{
                background-color: black;
                color: white;
            }
            .stTextArea>textarea {
                background-color: white;  /* White background */
                color: black;  /* Black text */
                border: 1px solid #ccc;  /* Light border */
            
            }
            .e17vllj40 {  /* This targets the deploy button */
                color: white !important;
            }
        </style>
    """, unsafe_allow_html=True)

st.title("Sentiment Analysis Of Movie Review")

st.write("""
    Enter a movie review and select a model to analyze the sentiment.
    
""")

# Accuracy scores for the models
# logreg_accuracy = accuracy_scores["logreg"]
# sgd_accuracy = accuracy_scores["sgd"]
# nb_accuracy = accuracy_scores["nb"]

st.header("Model Performance Comparision 🏆")
col1, col2, col3 = st.columns(3)

with col1:
    model_name = "Logistic Regression"
    # st.markdown(f"<h5>{model_name}</h5>", unsafe_allow_html=True)
    st.plotly_chart(create_gauge_chart(model_name, logreg_accuracy))


with col2:
    model_name = "SGD Classifier"
    # st.subheader(model_name)
    st.plotly_chart(create_gauge_chart(model_name, sgd_accuracy))
    

with col3:
    model_name = "Multinomial Naive Bayes"
    # st.subheader(model_name)
    st.plotly_chart(create_gauge_chart(model_name, nb_accuracy))
    

# Create a selectbox for the user to select the model
model_choice = st.sidebar.selectbox("Select a model", ["Logistic Regression", "SGD Classifier", "Multinomial Naive Bayes"])

user_input = st.text_area("Enter a review")
show_wordcloud = st.sidebar.radio("Show Word Cloud?", ["Yes", "No"])


# logreg_cm = confusion_matrices["logreg"]
# sgd_cm = confusion_matrices["sgd"]
# nb_cm = confusion_matrices["nb"]
if st.button('Analyze Sentiment'):
    if user_input:
        # Perform sentiment analysis
        with st.spinner('Analyzing your review...'):
            time.sleep(3)
            clean_user_input = clean_text(user_input)

            transformed_user_input = vectorizer.transform([clean_user_input])

            if model_choice == "Logistic Regression":
                model = lr_model
            elif model_choice == "SGD Classifier":
                model = sgd_model
            else:
                model = nb_model

            prediction = model.predict(transformed_user_input)[0]
            
            sentiment_map = {1: "Positive", 0: "Negative"}

            sentiment_text = sentiment_map[prediction]
        
            # Display sentiment prediction
            if sentiment_text == "Positive":
                st.write(f"Predicted Sentiment using {model_choice} 🧠: {sentiment_text} 😄🎉")
            else:
                st.write(f"Predicted Sentiment using {model_choice} 🧠: {sentiment_text} 😞💔")
            
            if show_wordcloud == 'Yes':
                st.write("Word Cloud of Your Review:")
                wordcloud_fig = generate_wordcloud(clean_user_input)
                st.pyplot(wordcloud_fig)

            st.write(f"Confusion Matrix for {model_choice} 📊💡")
            if model_choice == "Logistic Regression":
                st.plotly_chart(plot_confusion_matrix(logreg_cm, "Logistic Regression"))
            elif model_choice == "SGD Classifier":
                st.plotly_chart(plot_confusion_matrix(sgd_cm, "SGD Classifier"))
            else:
                st.plotly_chart(plot_confusion_matrix(nb_cm, "Naive Bayes"))

            
    else:
        st.write("Please enter a review to predict sentiment and generate the word cloud.")