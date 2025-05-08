import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
import spacy
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
from collections import Counter

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load NLP Models
try:
    nlp_spacy = spacy.load("en_core_web_sm")
    st.success("spaCy model loaded successfully")
except IOError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp_spacy = spacy.load("en_core_web_sm")
    st.success("spaCy model downloaded and loaded successfully")

summarizer = pipeline("summarization")

# Streamlit page configuration
st.set_page_config(page_title="🧠 NLP Analysis App", layout="wide")
st.title("📋 NLP Analysis Checklist App")

# File uploader
uploaded_file = st.file_uploader("📁 Upload CSV with a text column", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())  # Check if file is uploaded and readable
    text_column = st.selectbox("📌 Select the column containing text", df.columns)
    if text_column:
        st.write(df[text_column].head())  # Check if selected column contains valid text
        df['text'] = df[text_column].astype(str)

        # Cleaning function
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        def clean_text(text):
            text = text.lower()
            text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
            text = re.sub(r'\@w+|\#','', text)
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            tokens = word_tokenize(text)
            tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
            return " ".join(tokens)

        df['clean_text'] = df['text'].apply(clean_text)
        st.write(df[['text', 'clean_text']].head())  # Display cleaned text

        # Create tabs
        tabs = st.tabs([
            "📌 Goal", "🧹 Cleaning", "🔍 EDA", "😊 Sentiment", "🏷️ NER",
            "📚 Summarization", "📈 Visualize", "🧪 Evaluation"
        ])

        with tabs[0]:
            st.subheader("Define Your NLP Goal")
            goal = st.radio("Choose your goal:", [
                "Sentiment Analysis", "Text Classification", "Named Entity Recognition (NER)", "Summarization"
            ])
            st.success(f"You chose: **{goal}**")

        with tabs[1]:
            st.subheader("Cleaned Text Preview")
            st.dataframe(df[['text', 'clean_text']].head(10))

        with tabs[2]:
            st.subheader("Exploratory Data Analysis")
            word_list = " ".join(df['clean_text']).split()
            freq = Counter(word_list).most_common(30)
            freq_df = pd.DataFrame(freq, columns=['Word', 'Frequency'])

            fig, ax = plt.subplots()
            sns.barplot(x='Frequency', y='Word', data=freq_df, ax=ax)
            st.pyplot(fig)

            wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(word_list))
            st.image(wc.to_array(), caption="Word Cloud")

        with tabs[3]:
            st.subheader("Sentiment Analysis")
            analyzer = SentimentIntensityAnalyzer()

            def get_sentiment(text):
                blob = TextBlob(text)
                vader_score = analyzer.polarity_scores(text)['compound']
                return pd.Series([blob.sentiment.polarity, vader_score])

            df[['TextBlob_Sentiment', 'VADER_Sentiment']] = df['text'].apply(get_sentiment)
            st.write(df[['text', 'TextBlob_Sentiment', 'VADER_Sentiment']].head(10))

            fig, ax = plt.subplots()
            sns.histplot(df['TextBlob_Sentiment'], kde=True, label="TextBlob", color='blue')
            sns.histplot(df['VADER_Sentiment'], kde=True, label="VADER", color='orange')
            plt.legend()
            st.pyplot(fig)

        with tabs[4]:
            st.subheader("Named Entity Recognition (NER)")
            sample_text = st.text_area("Enter a sentence for NER", "Apple was founded by Steve Jobs in California.")
            doc = nlp_spacy(sample_text)
            for ent in doc.ents:
                st.write(f"• **{ent.text}** → {ent.label_}")

        with tabs[5]:
            st.subheader("Summarization")
            summary_text = st.text_area("Paste article or long paragraph:")
            if st.button("Generate Summary"):
                if len(summary_text) < 50:
                    st.warning("Text too short to summarize.")
                else:
                    summary = summarizer(summary_text, max_length=100, min_length=30, do_sample=False)
                    st.success(summary[0]['summary_text'])

        with tabs[6]:
            st.subheader("Visualization")

            st.write("📊 Sentiment Over Time (if date column exists):")
            date_col = st.selectbox("Optional: Select a date column", ["None"] + list(df.columns))
            if date_col != "None":
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                time_df = df.dropna(subset=[date_col])
                st.line_chart(time_df[[date_col, 'VADER_Sentiment']].set_index(date_col).resample('W').mean())

        with tabs[7]:
            st.subheader("Evaluation Metrics Guide")
            st.markdown(""" 
            **For Classification:**
            - Accuracy, Precision, Recall, F1-score

            **For Summarization:**
            - ROUGE / BLEU

            **For Topic Modeling:**
            - Coherence Score

            Use appropriate libraries like `sklearn.metrics`, `nltk.translate.bleu_score`, or `gensim.models.coherencemodel`.
            """)

else:
    st.info("Please upload a CSV file to start your NLP journey.")
