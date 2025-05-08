import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
import string

# App title
st.set_page_config(page_title="📰 News Sentiment Analyzer", layout="wide")
st.title("📰 News Sentiment Analyzer")

# Load data function
@st.cache_resource
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['text'] = df['text'].astype(str)
    return df

# Text cleaner
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return text

# Upload file
uploaded_file = st.file_uploader("📂 Upload your `reduced_news_data.csv` file", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)
    df['clean_text'] = df['text'].apply(clean_text)
    df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📌 Overview", "🧹 Genres with Text Cleaning", "📈 Sentiment & Length Analysis",
        "🔡 Word Frequency Comparison", "🔠 Top Words by Subject"
    ])

    with tab1:
        st.header("📌 Overview")
        st.subheader("Dataset Preview")
        st.dataframe(df.head(100))

        st.subheader("🧾 Summary")
        st.write(f"Total Articles: {len(df)}")
        st.write(f"Date Range: {df['date'].min().date()} to {df['date'].max().date()}")

        st.subheader("📊 Articles Over Time")
        time_data = df['date'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(10, 4))
        time_data.plot(ax=ax)
        ax.set_title("Articles Published Over Time")
        ax.set_ylabel("Number of Articles")
        st.pyplot(fig)

        st.subheader("📌 Top Subjects")
        top_subjects = df['subject'].value_counts().nlargest(10)
        fig, ax = plt.subplots()
        sns.barplot(y=top_subjects.index, x=top_subjects.values, ax=ax)
        ax.set_title("Top 10 Subjects")
        ax.set_xlabel("Count")
        st.pyplot(fig)

    with tab2:
        st.header("🧹 Cleaned Text View")
        st.dataframe(df[['subject', 'clean_text']].head(100))

    with tab3:
        st.header("📈 Sentiment & Word Count Distribution")
        st.subheader("📌 Word Count Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['word_count'], bins=50, kde=True, ax=ax)
        ax.set_title("Distribution of Article Length (in words)")
        ax.set_xlabel("Word Count")
        st.pyplot(fig)

    with tab4:
        st.header("🔡 Word Frequency by Genre")
        genre = st.selectbox("Choose Genre", df['subject'].dropna().unique())
        words = " ".join(df[df['subject'] == genre]['clean_text'].dropna())
        word_freq = Counter(words.split()).most_common(30)
        freq_df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])

        fig, ax = plt.subplots()
        sns.barplot(x='Frequency', y='Word', data=freq_df, ax=ax)
        ax.set_title(f"Top Words in {genre}")
        st.pyplot(fig)

    with tab5:
        st.header("🔠 Top Words by Subject (Word Cloud)")
        subject = st.selectbox("Choose a subject", df['subject'].unique(), key="subject_wordcloud")
        text_data = " ".join(df[df['subject'] == subject]['clean_text'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)
        st.image(wordcloud.to_array(), use_column_width=True)

else:
    st.warning("⚠️ Please upload your `reduced_news_data.csv` file to begin analysis.")
