import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from textblob import TextBlob
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, STOPWORDS
import spacy
from gensim import corpora
from gensim.models import LdaModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize


# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')  # For VADER sentiment analysis

# Load spaCy model (if not already loaded)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    !python -m spacy download en_core_web_sm
    nlp = spacy.load("en_core_web_sm")

# --- Define functions ---

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(tokens)

def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# --- Streamlit app ---

st.title("Text Analysis App")

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Sentiment Analysis", "Word Cloud", "Word Frequency",
                                            "Named Entity Recognition", "Topic Modeling", 
                                            "TF-IDF & Logistic Regression"])

# Global stop words (used across multiple tabs)
stop_words = set(stopwords.words('english'))

# --- Tab 1: Sentiment Analysis ---
with tab1:
    text_input = st.text_area("Enter text for analysis:", "")

    if st.button("Analyze"):
        # Clean the text
        cleaned_text = clean_text(text_input)

        # Analyze sentiment
        polarity = analyze_sentiment(cleaned_text)

        # Display results
        st.write("**Sentiment Polarity:**", polarity)

        if polarity > 0:
            st.write("**Sentiment:** Positive")
        elif polarity < 0:
            st.write("**Sentiment:** Negative")
        else:
            st.write("**Sentiment:** Neutral")

# --- Tab 2: Word Cloud ---
with tab2:
    text_input_wc = st.text_area("Enter text for word cloud:", "")

    if st.button("Generate Word Cloud"):
        # Clean the text
        cleaned_text_wc = clean_text(text_input_wc)

        # Generate and display word cloud 
        if len(cleaned_text_wc) > 0:
            wordcloud = WordCloud(width=800, height=400, 
                                  background_color='white', 
                                  stopwords=stop_words, 
                                  min_font_size=10).generate(cleaned_text_wc)
            
            plt.figure(figsize=(10, 7), facecolor=None)
            plt.imshow(wordcloud)
            plt.axis("off")
            plt.tight_layout(pad=0)
            plt.title("Word Cloud", fontsize=20)
            st.pyplot(plt)

# --- Tab 3: Word Frequency ---
with tab3:
    text_input_wf = st.text_area("Enter text for word frequency analysis:", "")

    if st.button("Analyze Word Frequency"):
        # Clean the text
        cleaned_text_wf = clean_text(text_input_wf)

        # Word Frequency Analysis
        all_words = cleaned_text_wf.split()
        freq_dist = nltk.FreqDist(all_words)
        common_words = freq_dist.most_common(15)  # Get top 15 words
        common_df = pd.DataFrame(common_words, columns=["Word", "Frequency"])

        # Display word frequency table
        st.write("**Top 15 Words:**")
        st.dataframe(common_df)

        # Plot word frequency
        plt.figure(figsize=(10, 5))
        sns.barplot(data=common_df, x='Frequency', y='Word', palette='viridis')
        plt.title("Top 15 Words")
        st.pyplot(plt)

# --- Tab 4: Named Entity Recognition ---
with tab4:
    text_input_ner = st.text_area("Enter text for NER:", "")

    if st.button("Run NER"):
        # Process text with spaCy
        doc = nlp(text_input_ner)

        # Display named entities
        st.write("**Named Entities:**")
        for ent in doc.ents:
            st.write(f"{ent.text} - {ent.label_}")

# --- Tab 5: Topic Modeling (LDA) ---
with tab5:
    text_input_lda = st.text_area("Enter text for topic modeling:", "")

    if st.button("Run Topic Modeling"):
        # Clean the text
        cleaned_text_lda = clean_text(text_input_lda)

        # Prepare data for LDA
        texts = [cleaned_text_lda.split()]  # Assuming single document for now
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]

        # Run LDA
        lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, passes=10)

        # Display topics
        st.write("**LDA Topics:**")
        for topic in lda_model.print_topics():
            st.write(topic)

# --- Tab 6: TF-IDF & Logistic Regression ---
with tab6:
    st.write("This tab would require training data and a pre-trained model for demonstration.")
    st.write("For simplicity, it's excluded from this example.")
    # You would add code here to load your data, train/load a model, 
    # and make predictions on new text input.
