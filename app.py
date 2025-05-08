import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re

from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Download necessary resources
nltk.download('punkt')
nltk.download('stopwords')

# ====== Streamlit App ======
st.set_page_config(layout="wide")
st.title("📰 Fake vs Real News - NLP Analysis")

# ====== Upload Data ======
uploaded_file = st.file_uploader("Upload reduced_news_data.csv", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.dropna(subset=['text'], inplace=True)
    df['label'] = df['subject'].apply(lambda x: 'Real' if x.lower() == 'real' else 'Fake')

    # ====== Clean Text ======
    stop_words = set(stopwords.words('english'))

    def clean_text(text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"[^a-z\s]", "", text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        return " ".join(tokens)

    df['clean_text'] = df['text'].apply(clean_text)

    st.subheader("🔍 Dataset Preview")
    st.dataframe(df[['title', 'label', 'clean_text']].head())

    # ====== Word Cloud ======
    st.subheader("☁️ Word Cloud")
    all_words = " ".join(df['clean_text'])
    wc = WordCloud(width=800, height=400, background_color='white').generate(all_words)
    fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
    ax_wc.imshow(wc, interpolation='bilinear')
    ax_wc.axis('off')
    st.pyplot(fig_wc)

    # ====== Sentiment Analysis ======
    st.subheader("😊 Sentiment Analysis")

    analyzer = SentimentIntensityAnalyzer()
    df['vader'] = df['clean_text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    df['textblob'] = df['clean_text'].apply(lambda x: TextBlob(x).sentiment.polarity)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**VADER Sentiment Distribution**")
        fig_vader, ax1 = plt.subplots()
        sns.histplot(df['vader'], kde=True, ax=ax1, color="skyblue")
        st.pyplot(fig_vader)

    with col2:
        st.markdown("**TextBlob Sentiment Distribution**")
        fig_tb, ax2 = plt.subplots()
        sns.histplot(df['textblob'], kde=True, ax=ax2, color="lightgreen")
        st.pyplot(fig_tb)

    # ====== Average Sentiment by Label ======
    st.subheader("📊 Average Sentiment by News Type")
    avg_sentiment = df.groupby('label')[['vader', 'textblob']].mean().reset_index()
    st.dataframe(avg_sentiment)

else:
    st.info("Please upload the `reduced_news_data.csv` file to begin.")
