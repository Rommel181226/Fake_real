import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from collections import Counter

# Function to load data
@st.cache_resource
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Convert date column
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
    
    # Check if the necessary columns exist
    if 'text' not in df.columns or 'subject' not in df.columns:
        st.error("The dataset must contain 'text' and 'subject' columns.")
    else:
        # Clean the text data
        df['clean_text'] = df['text'].astype(str).apply(clean_text)

        # Proceed with tabs and data processing after checking columns
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📌 Overview", "📚 Visualizing Genres", "🧹 Genres with Text Cleaning",
            "🔡 Word Frequency Comparison", "🔠 Top Words by Subject"
        ])

        with tab1:
            st.header("Overview")
            st.dataframe(df.head(100))  # Display the first 100 rows of the dataframe
            st.write(f"🧾 Total Articles: {len(df)}")
            st.write(f"📅 Date Range: {df['date'].min()} to {df['date'].max()}")
            # Add some additional data results in this tab
            st.write("### Summary Statistics")
            st.write(df.describe())

        with tab2:
            st.header("Visualizing Genres")
            subject_counts = df['subject'].value_counts()
            fig, ax = plt.subplots()
            sns.barplot(y=subject_counts.index, x=subject_counts.values, ax=ax)
            ax.set_title("Article Count by Genre")
            ax.set_xlabel("Count")
            st.pyplot(fig)
            # Display the counts for genres here
            st.write("### Genre Counts")
            st.write(subject_counts)

            # Get top 10 most frequent words for the dataset
            st.write("### Top 10 Most Frequent Words in the Dataset")
            words = " ".join(df['clean_text'].dropna())
            word_freq = Counter(words.split()).most_common(10)
            freq_df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
            st.write(freq_df)

        with tab3:
            st.header("Genres with Text Cleaning")
            st.write(df[['subject', 'clean_text']].head(100))
            # Show the cleaned text for some of the records
            st.write("### Sample Cleaned Text")
            st.write(df[['subject', 'clean_text']].head(10))

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
            # Display word frequency data
            st.write(f"### Top Words in {genre}")
            st.write(freq_df)

        with tab5:
            st.header("Top Words by Subject")
            subject = st.selectbox("Choose a subject", df['subject'].unique(), key="subject_topwords")
            text_data = " ".join(df[df['subject'] == subject]['clean_text'])
            wordcloud = WordCloud(width=800, height=400).generate(text_data)
            st.image(wordcloud.to_array())
            # Show a list of the top 10 words for the selected subject
            word_freq_subject = Counter(text_data.split()).most_common(10)
            word_freq_df = pd.DataFrame(word_freq_subject, columns=['Word', 'Frequency'])
            st.write(f"### Top 10 Words in {subject}")
            st.write(word_freq_df)

        # Adding a new tab for Subject Comparison (Fake vs. Real)
        tab6 = st.container()
        with tab6:
            st.header("Subject Comparison: Fake vs. Real News")
            
            # Assuming 'label' column distinguishes between fake and real news
            if 'label' not in df.columns:
                st.warning("The dataset does not have a 'label' column to distinguish fake and real news.")
            else:
                fake_news = df[df['label'] == 'fake']
                real_news = df[df['label'] == 'real']

                # Compare top subjects for fake and real news
                fake_subjects = fake_news['subject'].value_counts().nlargest(10)
                real_subjects = real_news['subject'].value_counts().nlargest(10)

                fig, ax = plt.subplots(figsize=(10, 6))
                fake_subjects.plot(kind='bar', color='red', alpha=0.5, label='Fake')
                real_subjects.plot(kind='bar', color='blue', alpha=0.5, label='Real')
                ax.set_title('Top 10 Subjects: Fake vs. Real News')
                ax.set_xlabel('Subjects')
                ax.set_ylabel('Frequency')
                ax.legend()
                st.pyplot(fig)

                # Display the top subjects data
                st.write("### Fake News Subjects")
                st.write(fake_subjects)
                st.write("### Real News Subjects")
                st.write(real_subjects)

else:
    st.warning("Please upload your `reduced_news_data.csv` file to begin.")
