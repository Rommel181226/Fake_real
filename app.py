import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from collections import Counter

# Set up the page configuration
st.set_page_config(page_title="News Sentiment Analyzer", layout="wide")

# Function to load data
@st.cache_resource
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df

# Function to clean text
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Keep only alphabets and spaces
    return text.lower()

# Title for the Streamlit app
st.title("📰 News Sentiment Analyzer")

# File uploader widget to upload CSV file
uploaded_file = st.file_uploader("📂 Upload your `reduced_news_data.csv` file", type=["csv"])

if uploaded_file:
    # Load and clean the data
    df = load_data(uploaded_file)
    df['clean_text'] = df['text'].astype(str).apply(clean_text)

    # Set up tabs for the app interface
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📌 Overview", "📚 Visualizing Genres", "🧹 Genres with Text Cleaning",
        "🔡 Word Frequency Comparison", "📊 Sentiment Distribution",
        "📈 Sentiment Comparison", "🔠 Top Words by Subject", "⚖️ Subject-wise Sentiment Comparison"
    ])

    with tab1:
        st.header("Overview")
        st.dataframe(df.head(100))  # Display the first 100 rows of the dataframe
        st.write(f"🧾 Total Articles: {len(df)}")
        st.write(f"📅 Date Range: {df['date'].min()} to {df['date'].max()}")

    with tab2:
        st.header("Visualizing Genres")
        subject_counts = df['subject'].value_counts()
        fig, ax = plt.subplots()
        sns.barplot(y=subject_counts.index, x=subject_counts.values, ax=ax)
        ax.set_title("Article Count by Genre")
        ax.set_xlabel("Count")
        st.pyplot(fig)

    with tab3:
        st.header("Genres with Text Cleaning")
        st.write(df[['subject', 'clean_text']].head(100))

    with tab4:
        st.header("Word Frequency Comparison")
        genre = st.selectbox("Choose Genre", df['subject'].dropna().unique())
        words = " ".join(df[df['subject'] == genre]['clean_text'].dropna())
        word_freq = Counter(words.split()).most_common(30)
        freq_df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])

        fig, ax = plt.subplots()
        sns.barplot(x='Frequency', y='Word', data=freq_df, ax=ax)
        ax.set_title(f"Top Words in {genre}")
        st.pyplot(fig)

    with tab5:
        st.header("Sentiment Distribution")
        sentiment_counts = df['label'].value_counts()
        fig, ax = plt.subplots()
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="pastel", ax=ax)
        ax.set_title("Sentiment Counts")
        st.pyplot(fig)

    with tab6:
        st.header("Sentiment Comparison by Genre")
        group = df.groupby(['subject', 'label']).size().unstack(fill_value=0)
        st.bar_chart(group)

    with tab7:
        st.header("Top Words by Subject")
        subject = st.selectbox("Choose a subject", df['subject'].unique(), key="subject_topwords")
        text_data = " ".join(df[df['subject'] == subject]['clean_text'])
        wordcloud = WordCloud(width=800, height=400).generate(text_data)
        st.image(wordcloud.to_array())

    with tab8:
        st.header("Subject-wise Sentiment Comparison")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=df, x='subject', hue='label', palette='Set2', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_title("Sentiment by Subject")
        st.pyplot(fig)

else:
    st.warning("Please upload your `reduced_news_data.csv` file to begin.")
