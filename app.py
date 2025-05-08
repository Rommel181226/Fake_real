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

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load NLP Models
nlp_spacy = spacy.load("en_core_web_sm")
summarizer = pipeline("summarization")

st.set_page_config(page_title="🧠 NLP Analysis App", layout="wide")
st.title("📋 NLP Analysis Checklist App")

# File uploader
uploaded_file = st.file_uploader("📁 Upload CSV with a text column", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Ensure the user selects the correct text column
    if df.empty:
        st.error("The uploaded file is empty. Please upload a valid CSV.")
    else:
        text_column = st.selectbox("📌 Select the column containing text", df.columns)
        if text_column:
            df['text'] = df[text_column].astype(str)
        else:
            st.error("No valid text column selected. Please check your file.")
    
        # Cleaning
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        def clean_text(text):
            if not text:
                return ""
            text = text.lower()
            text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
            text = re.sub(r'\@w+|\#','', text)
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            tokens = word_tokenize(text)
            tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
            return " ".join(tokens)

        df['clean_text'] = df['text'].apply(clean_text)

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
                if len(text.split()) < 3:  # Check for very short text
                    return pd.Series([None, None])
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
            if sample_text:
                doc = nlp_spacy(sample_text)
                if doc.ents:
                    for ent in doc.ents:
                        st.write(f"• **{ent.text}** → {ent.label_}")
                else:
                    st.warning("No named entities found.")
        
        with tabs[5]:
            st.subheader("Summarization")
            summary_text = st.text_area("Paste article or long paragraph:")
            if st.button("Generate Summary"):
                if len(summary_text) < 50:
                    st.warning("Text too short to summarize.")
                else:
                    try:
                        summary = summarizer(summary_text, max_length=100, min_length=30, do_sample=False)
                        st.success(summary[0]['summary_text'])
                    except Exception as e:
                        st.error(f"Error during summarization: {str(e)}")

        with tabs[6]:
            st.subheader("Visualization")
            st.write("📊 Sentiment Over Time (if date column exists):")
            date_col = st.selectbox("Optional: Select a date column", ["None"] + list(df.columns))
            if date_col != "None":
                df[date_col] = pd.to
