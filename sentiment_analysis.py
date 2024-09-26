import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import pandas as pd
from bs4 import BeautifulSoup
import re
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import streamlit as st
import yaml
from nrclex import NRCLex

# Load configuration
from src.config.config import DATA_FILE, RELEVANT_COLUMNS, MODEL_NAME

# with open('./src/config/config.yaml', 'r') as file:
#     config = yaml.safe_load(file)

# DATA_FILE = config['data_file']
# RELEVANT_COLUMNS = config['relevant_columns']
# SAVE_DIRECTORY = config['save_directory']
# MODEL_NAME = config['model_name']


# Load model and tokenizer from Hugging Face model hub cached)
@st.cache_resource
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer(MODEL_NAME)


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
    
    
def detect_emotions(text):
    emotion_analyzer = NRCLex(text)
    top_emotions = emotion_analyzer.top_emotions
    return {emotion[0]: emotion[1] for emotion in top_emotions}


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
        
        # emotion detection
        emotions = detect_emotions(text)

        # Return results
        return {
            'predicted_label': predicted_label,
            'predicted_index': predicted_index + 1, 
            'confidence_percentage': confidence_percentage,
            'emotions': emotions 
        }

    except Exception as e:
        return {
            'error': str(e)
        }


# Apply sentiment scoring to a DataFrame
def apply_sentiment_scoring(df):
    results = df['processed_feedback'].apply(sentiment_score)

    # Extract results into separate Series
    df['sentiments'] = results.apply(lambda x: x.get('predicted_label'))
    df['sentiments_index'] = results.apply(lambda x: x.get('predicted_index'))
    df['percentage_confidence'] = results.apply(lambda x: x.get('confidence_percentage'))

    return df

def get_keywords(text, result):
    words = text.split()
    if result['predicted_label'] == 'Positive':
        top_words = sorted(result['emotions'].items(), key=lambda x: x[1], reverse=True)[:3]
        return ', '.join([word[0] for word in top_words])
    elif result['predicted_label'] == 'Negative':
        bottom_words = sorted(result['emotions'].items(), key=lambda x: x[1])[:3]
        return ', '.join([word[0] for word in bottom_words])
    else:
        return "Neutral sentiment, no specific keywords identified"

def interpret_emotions(emotions):
    """
    Interprets the detected emotions and returns a user-friendly message.

    Args:
        emotions: A dictionary containing detected emotions and their scores.

    Returns:
        A string message describing the most prominent emotions.
    """
    
    # Filter out emotions with zero scores
    non_zero_emotions = {emotion: score for emotion, score in emotions.items() if score > 0}
    
    if not non_zero_emotions:
        return "No specific emotions were detected in your feedback."

    # Sort emotions by score
    sorted_emotions = sorted(non_zero_emotions.items(), key=lambda x: x[1], reverse=True)
    
    # Take top 3 emotions
    top_emotions = sorted_emotions[:3]
    
    if len(top_emotions) == 1:
        return f"Your feedback seems to express {top_emotions[0][0]}."
    else:
        emotion_list = ", ".join([emotion[0] for emotion in top_emotions[:-1]])
        last_emotion = top_emotions[-1][0]
        return f"Your feedback seems to express {emotion_list}, and {last_emotion}."

# Add a get_explanation function
def get_explanation(text, result):
    emotion_explanation = interpret_emotions(result['emotions'])
    return f"Based on the emotions detected, {emotion_explanation} This aligns with the overall sentiment of '{result['predicted_label']}'."
