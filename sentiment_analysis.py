import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import pandas as pd
from bs4 import BeautifulSoup
import re
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import streamlit as st
import yaml

# Load configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

DATA_FILE = config['data_file']
RELEVANT_COLUMNS = config['relevant_columns']
MODEL_NAME = config['model_name']
# SAVE_DIRECTORY = config['save_directory']

# Load model and tokenizer from Hugging Face model hub cached)
@st.cache_resource
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

# Load from local directory (if you have a fine-tuned model)
# @st.cache_resource
# def load_model_and_tokenizer(save_directory):
#     tokenizer = AutoTokenizer.from_pretrained(save_directory)
#     model = AutoModelForSequenceClassification.from_pretrained(save_directory)
#     return model, tokenizer

# model, tokenizer = load_model_and_tokenizer(SAVE_DIRECTORY) 


# Preprocessing function
def preprocess_text(text):
    """
    Preprocess a text string for sentiment analysis.

    Parameters
    ----------
    text : str
        The text string to preprocess.

    Returns
    -------
    str
        The preprocessed text string.
    """

    # Define the denoise_text function
    def denoise_text(text):
        text = strip_html(text)
        return text

    # Define the strip_html function
    def strip_html(text):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    # Apply denoising functions
    text = denoise_text(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs, hashtags, mentions, and special characters
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)

    # Remove numbers/digits
    text = re.sub(r'\b[0-9]+\b\s*', '', text)

    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join tokens back into a single string
    return ' '.join(tokens)
    

# Sentiment scoring function
def sentiment_score(text, model=model, tokenizer=tokenizer, label_mapping={1: 'Negative', 2: 'Neutral', 3: 'Positive'}): 
    try:
        # Tokenize the input text
        tokens = tokenizer.encode(text, return_tensors='pt')

        # Get model predictions
        with torch.no_grad():
            result = model(tokens)

        # Obtain predicted class index (no increment)
        predicted_index = torch.argmax(result.logits).item()
        
        # Map scores to labels
        if label_mapping is not None:
            predicted_label = label_mapping.get(predicted_index + 1, f'Class {predicted_index + 1}')

        # Map scores to labels
        # predicted_label = label_mapping.get(predicted_index, f'Class {predicted_index}') 

        # Calculate confidence percentage
        probabilities = softmax(result.logits, dim=1)
        confidence_percentage = str(probabilities[0, predicted_index].item() * 100) + '%'

        # Return results
        return {
            'predicted_label': predicted_label,
            'predicted_index': predicted_index + 1, 
            'confidence_percentage': confidence_percentage
        }

    except Exception as e:
        return {
            'error': str(e)
        }


# Apply sentiment scoring to a DataFrame (corrected)
# def apply_sentiment_scoring(df):
#     # Apply sentiment scoring 
#     df[['sentiments', 'sentiments_index', 'percentage_confidence']] = df['processed_feedback'].apply(sentiment_score)

#     return df

# Apply sentiment scoring to a DataFrame
def apply_sentiment_scoring(df):
    results = df['processed_feedback'].apply(sentiment_score)

    # Extract results into separate Series
    df['sentiments'] = results.apply(lambda x: x.get('predicted_label'))
    df['sentiments_index'] = results.apply(lambda x: x.get('predicted_index'))
    df['percentage_confidence'] = results.apply(lambda x: x.get('confidence_percentage'))

    return df
