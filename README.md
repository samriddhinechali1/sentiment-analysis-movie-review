# Movie Review Sentiment Analysis 🎬

This app takes movie reviews, analyzes them, and tells you if they're positive or negative. It uses machine learning models like **Logistic Regression**, **SGD Classifier**, and **Naive Bayes** to predict sentiment from the text you provide. 

## 🚀 Features
- **Sentiment Prediction**: Enter a movie review, and we'll tell you if it's positive or negative.
- **Word Clouds**: Want to see what words pop up most in your review? We’ve got you covered with a fun word cloud visualization.
- **Model Comparison**: Wondering which model works best? We’ve got performance comparisons to help you decide.
- **Multiple Themes**: Dark mode or light mode? Choose your preferred theme while using the app!

## ⚙️ How it Works
This app uses natural language processing (NLP) and machine learning to analyze movie reviews. It preprocesses the text, vectorizes it using **TF-IDF**, and then feeds it into models like **Logistic Regression**, **SGD Classifier**, and **Naive Bayes** to predict sentiment. The models are trained on a **sampled subset** of a large dataset containing 50,000 movie reviews. 

But **beware**... with great power comes great responsibility (and some slow loading times). 🚨

## 💡 Troubles Faced During Development
- **Dataset Size**: The initial dataset was massive (50,000 movie reviews). Trying to load and process all that data caused some performance hiccups so we sampled the dataset. Even though the app works, it can run a bit slow when processing large chunks of text or generating word clouds. **Patience is key**! ⏳
- **Model Size**: The models themselves, especially when trained on TF-IDF vectors, were **huge**! Storing and managing these models took up way more space than expected. We had to get creative with managing the model storage.
- **File Storage**: Saving models as `.pkl` files was too big for GitHub, so we had to rely on some other solutions.

## ⚠️ Known Issues
- **Slow Performance**: Due to the large dataset, **the app may run a little slow**, especially if you're generating word clouds. The machine learning models take a bit of time to load and process, so please be patient.
- **No Caching**: We haven't implemented caching for faster model inference yet (that's a future upgrade!). So every time the app loads or a prediction is made, it goes through the whole process. But hey, it’s all part of the fun! 🕶️

## 🌟 How to Run the App Locally
1. Clone this repository:
   ```bash
   git clone https://github.com/samriddhinechali1/movie-review-sentiment-analysis.git
   ```
2. Run the app:
   ```bash
   streamlit run review_app.py
   ```
## 🛠️ Technologies Used
* **Streamlit**: For creating the interactive web app.
* **Scikit-Learn**: For building and evaluating machine learning models.
* **Plotly & Matplotlib**: For creating performance and visualization charts (because who doesn’t love a good graph?).
* **NLTK**: For text preprocessing and sentiment analysis.
* **Pandas & NumPy**: For data manipulation and numerical operations.
## 🎉 Final Thoughts
This app may not be the fastest, but it certainly gets the job done when it comes to predicting sentiment! It was an incredibly fun project, and although there were plenty of trial and error moments, I learned a lot along the way. The app is perfect for anyone curious to see if their favorite movie will leave them with a smile or feeling disappointed.

While the app could still be optimized for better performance, it's fully functional and serves its purpose of providing sentiment analysis for movie reviews. If you have suggestions on improving speed, functionality, or have any other ideas, feel free to open an issue or submit a pull request—contributions are always welcome!


