# EMAIL-SMS_Classifier_Using_ML-and-NLP

A Machine Learning & NLP-based project to classify emails/SMS messages as Spam or Ham.
Built with TF-IDF, Multinomial Naive Bayes, and deployed as a simple interactive web app.

🚀 Project Overview

The goal of this project is to build a spam detection system that can effectively classify text messages as spam (unwanted/promotional/fraud) or ham (legitimate messages).

Data imported from Kaggle SMS Spam Collection dataset

Performed Exploratory Data Analysis (EDA) and data cleaning

Applied Natural Language Processing (NLP) for text preprocessing

Used TF-IDF vectorization to convert text into numerical features

Trained a Multinomial Naive Bayes model achieving ~95% accuracy

Deployed the model with Streamlit (SHIELD-IT web app)

⚙️ Workflow

Data Collection → Kaggle SMS Spam Dataset

EDA & Cleaning → Removed duplicates, punctuation, stopwords, lowercased text

NLP Processing → Tokenization, stemming, TF-IDF vectorization

Modeling → Multinomial Naive Bayes classifier

Evaluation → Accuracy, confusion matrix (~95% accuracy)

Deployment → Streamlit app where user enters text and gets prediction

🛠️ Tech Stack

Languages: Python

Libraries: pandas, numpy, scikit-learn, nltk, streamlit

ML Model: Multinomial Naive Bayes

Vectorizer: TF-IDF

Deployment: Streamlit (SHIELD-IT app)


🔧 Installation & Setup

Clone the repository

git clone https://github.com/your-username/SHIELD-IT-Spam-Classifier.git
cd SHIELD-IT-Spam-Classifier


Install dependencies

pip install -r requirements.txt


Run the app

streamlit run app.py

🌐 Deployment

The model is deployed with Streamlit under the name SHIELD-IT.

Enter any email or SMS

Get instant prediction: Spam ❌ or Ham ✅

📊 Results

Accuracy: ~95%

Model: Multinomial Naive Bayes

Strengths: Simple, fast, effective for text classification tasks

📌 Future Improvements

Add deep learning models (LSTM/BERT) for better accuracy

Handle multilingual spam detection

Improve UI/UX of the web app

🙌 Acknowledgements

Dataset: Kaggle - SMS Spam Collection

Libraries: NLTK, Scikit-learn, Streamlit

✨ Developed with ❤️ using Machine Learning & NLP
