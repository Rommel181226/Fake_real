import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data (only once)
nltk.download("punkt")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# Text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    filtered = [word for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(filtered)

# Sentiment analysis using TextBlob
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("data/news_cleaned.csv")  # Make sure this file exists
    df.dropna(subset=["text"], inplace=True)
    df["clean_text"] = df["text"].apply(clean_text)
    df["Sentiment"] = df["clean_text"].apply(analyze_sentiment)
    return df

# Streamlit App
st.set_page_config(page_title="News Analyzer", layout="wide")
st.title("📰 News Sentiment & Genre Analysis")

# Load dataset
df = load_data()

# Sidebar filters
st.sidebar.header("Filter Articles")
subjects = df["subject"].unique().tolist()
selected_subjects = st.sidebar.multiselect("Select genres:", subjects, default=subjects)

filtered_df = df[df["subject"].isin(selected_subjects)]

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Sentiment", "🧾 Raw Data"])

with tab1:
    st.subheader("Articles per Genre")
    genre_count = filtered_df["subject"].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=genre_count.index, y=genre_count.values, palette="Set2", ax=ax)
    plt.xticks(rotation=45)
    ax.set_ylabel("Number of Articles")
    st.pyplot(fig)

with tab2:
    st.subheader("Sentiment Distribution")
    sentiment_count = filtered_df["Sentiment"].value_counts()
    fig2, ax2 = plt.subplots()
    sns.barplot(x=sentiment_count.index, y=sentiment_count.values, palette="coolwarm", ax=ax2)
    ax2.set_ylabel("Number of Articles")
    st.pyplot(fig2)

with tab3:
    st.subheader("Dataset (Filtered)")
    st.dataframe(filtered_df[["title", "subject", "Sentiment", "date", "clean_text"]])

