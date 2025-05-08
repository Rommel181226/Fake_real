import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
import re
from wordcloud import WordCloud

# Set the page config for Netflix-like style
st.set_page_config(page_title="News Sentiment Analyzer", layout="wide")

# Apply dark theme (Streamlit custom theme)
st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: white;
        }
        .css-1v0mbdj {
            background-color: #121212 !important;
        }
        .stButton>button {
            background-color: #e50914;
            color: white;
            font-weight: bold;
        }
        .stSlider>div>div>div {
            background-color: #e50914;
        }
        .stTextInput input {
            background-color: #333;
            color: white;
        }
        .stSelectbox>div>div {
            background-color: #333;
        }
    </style>
""", unsafe_allow_html=True)

# Load your data here
@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    return df

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower()

# Title and File Upload
st.title("📰 News Sentiment Analyzer")
uploaded_file = st.file_uploader("📂 Upload your `reduced_news_data.csv` file", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)
    df['clean_text'] = df['text'].astype(str).apply(clean_text)

    # Horizontal Tabs for Navigation
    tab1, tab2, tab3, tab4, tab8 = st.tabs([
        "📌 Overview", 
        "📚 Visualizing Genres", 
        "🧹 Genres with Text Cleaning",
        "🔡 Word Frequency Comparison", 
        "⚖️ Subject-wise Sentiment Comparison"
    ])

    with tab1:
        st.header("Overview")
        st.dataframe(df.head(100))
        st.write(f"🧾 Total Articles: {len(df)}")

    with tab2:
        st.header("Visualizing Genres")
        subject_counts = df['subject'].value_counts()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(y=subject_counts.index, x=subject_counts.values, ax=ax, palette="dark:red")
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

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Frequency', y='Word', data=freq_df, ax=ax, palette="dark:red")
        ax.set_title(f"Top Words in {genre}")
        st.pyplot(fig)

    with tab8:
        st.header("Subject-wise Sentiment Comparison")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=df, x='subject', hue='label', palette='Set2', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_title("Sentiment by Subject")
        st.pyplot(fig)
else:
    st.warning("Please upload your `reduced_news_data.csv` file to begin.")
