import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import re
import string
import requests
import datetime
from bs4 import BeautifulSoup

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from torch.nn.functional import softmax
import torch

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

# Configure Streamlit page
st.set_page_config(
    page_title="SA App",
    page_icon="R.png",
    layout="wide",
)

st.title(" ")
st.markdown("<h1 style='text-align: center;'>Sentiment Analysis App</h1>", unsafe_allow_html=True)
st.title(" ")

# Add an introductory paragraph
st.markdown("""
This web application is a sentiment analysis tool developed by Raqib (popularly known as raqibcodes). It can be used to determine whether user-entered text has a Positive or Negative sentiment. The underlying text classification model was trained on feedback data collected from 300 level undergraduate Computer Engineering students at the University of Ilorin (who are Raqib's peers). Subsequently, the model underwent fine-tuning using BERT and KerasNLP techniques, resulting in an impressive accuracy score of 96%. The objective is to uncover sentiments expressed in the feedback and gain a comprehensive understanding of student perceptions and satisfaction regarding their educational experience.  
To utilize the application, simply input your text, and it will promptly reveal the underlying sentiment.
The app also has Exploratory Data Analysis capabilities.
""")

# Initialize session state
if 'course_code' not in st.session_state:
    st.session_state.course_code = None
if 'previous_exp' not in st.session_state:
    st.session_state.previous_exp = None
if 'gender' not in st.session_state:
    st.session_state.gender = None
if 'attendance' not in st.session_state:
    st.session_state.attendance = None
if 'difficulty' not in st.session_state:
    st.session_state.difficulty = None
if 'study_hours' not in st.session_state:
    st.session_state.study_hours = None
if 'satisfaction' not in st.session_state:
    st.session_state.satisfaction = None
if 'department' not in st.session_state:
    st.session_state.department = None

# Create containers for sideboxes
course_code_container = st.empty()
previous_exp_container = st.empty()
gender_container = st.empty()
attendance_container = st.empty()
difficulty_container = st.empty()
study_hours_container = st.empty()
satisfaction_container = st.empty()
department_container = st.empty()

# Unique identifier for each selectbox
selectbox_keys = ['course_code', 'previous_exp', 'gender', 'attendance', 'difficulty', 'study_hours', 'satisfaction', 'department']
# Get values from sideboxes
course_code = course_code_container.selectbox("Course Code", ['Select Course Code', 'CPE 321', 'CPE 311', 'CPE 341', 'CPE 381', 'CPE 331', 'MEE 361', 'GSE 301'], key=selectbox_keys[0])
previous_exp = previous_exp_container.selectbox("Previous Experience", ['Select Option', "Yes", "No"], key=selectbox_keys[1])
gender = gender_container.selectbox("Gender", ['Select Gender', 'Male', 'Female'], key=selectbox_keys[2])
attendance = attendance_container.selectbox("Attendance", ['Select Attendance', 'Regular', 'Irregular', 'Occasional'], key=selectbox_keys[3])
difficulty = difficulty_container.selectbox("Course Difficulty", ['Select Difficulty', 'Easy', 'Difficult', 'Challenging', 'Moderate'], key=selectbox_keys[4])
study_hours = st.select_slider("Study Hours (per week)", options=list(range(25)), key=selectbox_keys[5], value=None)
satisfaction = st.select_slider("Overall Satisfaction", options=list(range(1, 11)), key=selectbox_keys[6], value=None)
department = department_container.selectbox("Department", ['Select Option', "Yes", "No"], key=selectbox_keys[7])

# Load the exported data using st.cache
# @st.cache_data()
# def load_data():
#     return pd.read_csv('survey_data.csv')

# df = load_data()

df = pd.read_csv('survey_data.csv')
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')
# df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')
# df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

# Text Preprocessing of the texts column using NLTK
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
    
X_preprocessed = [preprocess_text(text) for text in df['feedback']]


# model name
model_name = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')
model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')
# # load directory of saved model
# save_directory = r"C:\Users\user\Desktop\MACHINE LEARNING\Sentiment Analysis\New folder"
# # load model from the local directory
# tokenizer = AutoTokenizer.from_pretrained(save_directory)
# model = AutoModelForSequenceClassification.from_pretrained(save_directory)


# calculate sentiment scoring
def sentiment_score(text, model, tokenizer, label_mapping={1: 'Negative', 2: 'Neutral', 3: 'Positive'}):
    try:
        # Tokenize the input text
        tokens = tokenizer.encode(text, return_tensors='pt')

        # Get model predictions
        with torch.no_grad():
            result = model(tokens)

        # Obtain predicted class index
        predicted_index = torch.argmax(result.logits).item()

        # Map scores to labels
        if label_mapping is not None:
            predicted_label = label_mapping.get(predicted_index + 1, f'Class {predicted_index + 1}')

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


# Function to apply sentiment scoring to a single feedback
def apply_sentiment_scoring(feedback):
    # Apply sentiment scoring
    result = sentiment_score(feedback, model, tokenizer, label_mapping={1: 'Negative', 2: 'Neutral', 3: 'Positive'})

    # Return the sentiment scoring results
    # return {
    #     'sentiments': result.get('predicted_label', None),
    #     'sentiments_index': result.get('predicted_index', None),
    #     'percentage_confidence': result.get('confidence_percentage', None)
    # }

    return pd.Series({
        'sentiments': result.get('predicted_label', None),
        'sentiments_index': result.get('predicted_index', None),
        'percentage_confidence': result.get('confidence_percentage', None)
    })


user_input = st.text_area("Enter Your Text Feedback Here:")
if st.button("Analyze Sentiment") and user_input:
    with st.spinner("Analyzing sentiment......."):
        # Retrieve values from the sideboxes only when there is user input
        course_code = st.session_state.course_code or 'Select Course Code'
        previous_exp = st.session_state.previous_exp
        gender = st.session_state.gender
        attendance = st.session_state.attendance
        difficulty = st.session_state.difficulty
        study_hours = st.session_state.study_hours
        satisfaction = st.session_state.satisfaction
        department = st.session_state.department

        # Preprocess the user input
        user_input_processed = preprocess_text(user_input)
        result = sentiment_score(user_input_processed, model, tokenizer, label_mapping={1: 'Negative', 2: 'Neutral', 3: 'Positive'})
        st.subheader("Sentiment Analysis Result:")
        st.write(f"Predicted Sentiment: {result.get('predicted_label', 'N/A')}")
        st.write(f"Sentiment Index: {result.get('predicted_index', 'N/A')}")
        st.write(f"Confidence Percentage: {result.get('confidence_percentage', 'N/A')}")

        # Update the DataFrame with the new feedback
        new_feedback = pd.DataFrame({
        'course code': [course_code], 
        'feedback': [user_input],
        'previous experience': [previous_exp],
        'gender': [gender],
        'attendance': [attendance],
        'course difficulty': [difficulty],
        'study hours (per week)': [study_hours],
        'overall satisfaction': [satisfaction],
        'department': [department],
         'date': [pd.to_datetime('now').date()],
        # 'date': [pd.to_datetime('now').date()], 
        'time': [pd.to_datetime('now').time()],
        # 'time': [datetime.datetime.now().strftime('%H:%M:%S')], 
        'hour': [pd.to_datetime('now').hour],
        })

        # Preprocess and add the new feedback to the DataFrame
        new_feedback['processed_feedback'] = new_feedback['feedback'].apply(preprocess_text)
        new_feedback['char_count'] = len(user_input_processed)
        new_feedback['word_count'] = len(user_input_processed.split())
        new_feedback[['sentiments', 'sentiments_index', 'percentage_confidence']] = new_feedback['processed_feedback'].apply(apply_sentiment_scoring)
        # new_feedback[['sentiments', 'sentiments_index', 'percentage_confidence']] = new_feedback.apply(apply_sentiment_scoring, axis=1)

        df['date'] = pd.to_datetime(df['date'])
        df = pd.concat([df, new_feedback], ignore_index=True)

        # Save the updated dataset to the CSV file
        try:
            df.to_csv('survey_data.csv', index=False)
                
        except Exception as e:
            st.error(f"Error saving data: {str(e)}")
                
        # Display success message
        st.success("Sentiment analysis completed!")
        # Add a download button for the CSV file
        st.download_button(
            label="Download Updated Dataset",
            data=df.to_csv(index=False).encode(),
            file_name='survey_data_updated.csv',
            mime='text/csv'
        )

st.markdown(
    f"""
    <style>
        div.stButton > button:first-child {{
            background-color: #636EFA;
            color: white;
            font-weight: bold;
            font-size: 18px;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# separate view for visualizations
if st.button("Explore Visualizations"):
    # Create a subpage for visualizations
    with st.expander("Sentiments Distribution"):
        sentiment_counts = df['sentiments'].value_counts()
        fig = px.bar(sentiment_counts, x=sentiment_counts.index, y=sentiment_counts.values, labels={'x': 'Sentiment', 'y': 'Count'})
        fig.update_layout(
            xaxis=dict(type='category'),
            title="Distribution of Sentiments",
            xaxis_title="Sentiment",
            yaxis_title="Count",
        )
        st.plotly_chart(fig)

    with st.expander("Pie Chart Distribution of Sentiments "):
        label_data = df['sentiments'].value_counts()
        fig = px.pie(label_data, values=label_data.values, names=label_data.index, hole=0.4)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(title="Sentiments Distribution (Pie Chart)")
        st.plotly_chart(fig)

    with st.expander("Word Cloud Visualization"):
        all_feedback = ' '.join(df['processed_feedback'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_feedback)
        fig = px.imshow(wordcloud)
        fig.update_layout(title='Word Cloud of Overall Feedback Text')
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        st.plotly_chart(fig)
        
    with st.expander("Course Difficulty"):
        course_difficulty_counts = df['course difficulty'].value_counts()
        fig = px.bar(course_difficulty_counts, x=course_difficulty_counts.index, y=course_difficulty_counts.values, labels={'x': 'Course Difficulty', 'y': 'Count'})
        fig.update_layout(
            xaxis=dict(type='category'),
            title="Feedback Count by Course Difficulty",
            xaxis_title="Course Difficulty",
            yaxis_title="Count",
        )
        st.plotly_chart(fig)

    with st.expander("Feedback Count by Course Code"):
        course_code_counts = df['course code'].value_counts()
        fig = px.bar(course_code_counts, x=course_code_counts.index, y=course_code_counts.values, labels={'x': 'Course Code', 'y': 'Count'})
        fig.update_layout(
            xaxis=dict(type='category'),
            title="Feedback Count by Course Code",
            xaxis_title="Course Code",
            yaxis_title="Count",
        )
        st.plotly_chart(fig)

    with st.expander("Gender Distribution"):
        gender_counts = df['gender'].value_counts()
        fig = px.pie(gender_counts, values=gender_counts.values, names=gender_counts.index, hole=0.4)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(title="Gender Distribution")
        st.plotly_chart(fig)
        
    with st.expander("Most Frequently Used Words"):
        from collections import Counter
        word_frequency = Counter(" ".join(df['feedback']).split()).most_common(30)
        word_df = pd.DataFrame(word_frequency, columns=['Word', 'Frequency'])
        fig = px.bar(word_df, x='Frequency', y='Word', orientation='h', title='Top 30 Most Frequently Used Words')
        st.plotly_chart(fig)
        
    with st.expander("Course Code distribution by Sentiment distribution"):
        fig = px.histogram(df, x='course code', color='sentiments', title='Course Code distribution by Sentiment distribution')
        fig.update_xaxes(title='Course Code')
        fig.update_yaxes(title='Count of Feedback')
        st.plotly_chart(fig)
        
    with st.expander("Sentiment Distribution by Course Difficulty"):
        fig = px.histogram(df, x='course difficulty', color='sentiments', 
                           title='Sentiment Distribution by Course Difficulty',
                           category_orders={"Course Difficulty": ['Easy', 'Moderate', 'Challenging', 'Difficult']})
        fig.update_xaxes(title='Course Difficulty')
        fig.update_yaxes(title='Count of Feedback')
        st.plotly_chart(fig)
        
    with st.expander("Sentiment Distribution by Gender"):
        fig = px.histogram(df, x='gender', color='sentiments', title='Sentiment Distribution by Gender')
        fig.update_xaxes(title='Gender')
        fig.update_yaxes(title='Count of Feedback')
        st.plotly_chart(fig)
        
    with st.expander("Distribution of Study Hours (per week) and Overall Satisfaction"):
        fig = px.scatter(df, x='study hours (per week)', y='overall satisfaction')
        fig.update_layout(
            title="Distribution of Study Hours (per week) and Overall Satisfaction",
            xaxis_title="Study Hours (per week)",
            yaxis_title="Overall Satisfaction",
        )
        st.plotly_chart(fig)

    with st.expander("Distribution of Study Hours by Sentiment"):
        fig = px.histogram(df, x='study hours (per week)', color='sentiments', 
                           title='Distribution of Study Hours by Sentiment')
        fig.update_xaxes(title='Study Hours per Week')
        fig.update_yaxes(title='Count of Feedback')
        st.plotly_chart(fig)

    with st.expander("Distribution of Word Count for different levels of Course Difficulty"):
        fig = px.box(df, x='course difficulty', y='word_count', 
                         title='Distribution of Word Count for different levels of Course Difficulty',
                         category_orders={"course difficulty": ['Easy', 'Moderate', 'Challenging', 'Difficult']})
        fig.update_xaxes(title='Course Difficulty')
        fig.update_yaxes(title='Word Count')
        st.plotly_chart(fig)

    with st.expander("Overall Satisfaction vs. Sentiment"):
        fig = px.box(df, x='sentiments', y='overall satisfaction', 
                         title='Overall Satisfaction vs. Sentiment')
        fig.update_xaxes(title='Sentiment')
        fig.update_yaxes(title='Overall Satisfaction')
        st.plotly_chart(fig)

    with st.expander("Sentiment Over Time"):
        fig = px.line(df, x='date', y='sentiments_index', 
                          title='Sentiment Over Time')
        fig.update_xaxes(title='Date')
        fig.update_yaxes(title='Sentiment Index')
        st.plotly_chart(fig)
   

        
# Add a checkbox to show/hide the DataFrame
show_df = st.checkbox("Show Data")

# Display the DataFrame only if the checkbox is selected
if show_df:
    st.write(df)
df['percentage_confidence'] = df['percentage_confidence'].apply(lambda x: float(x.strip('%')))

# Sentiment Summary
st.header("Sentiment Summary")
positive_feedback_count = (df["sentiments_index"] == 3).sum()
neutral_feedback_count = (df["sentiments_index"] == 2).sum()
negative_feedback_count = (df["sentiments_index"] == 1).sum()

# Calculate average confidence percentage for each sentiment
average_confidence_positive = df.loc[df["sentiments_index"] == 3, "percentage_confidence"].mean()
average_confidence_neutral = df.loc[df["sentiments_index"] == 2, "percentage_confidence"].mean()
average_confidence_negative = df.loc[df["sentiments_index"] == 1, "percentage_confidence"].mean()

# Display updated summary statistics
st.write(f"Number of Positive Feedback: {positive_feedback_count}")
st.write(f"Number of Neutral Feedback: {neutral_feedback_count}")
st.write(f"Number of Negative Feedback: {negative_feedback_count}")

# Display average confidence percentage for each sentiment
st.write(f"Average Confidence Percentage for Positive Feedback: {average_confidence_positive:.2f}%")
st.write(f"Average Confidence Percentage for Neutral Feedback: {average_confidence_neutral:.2f}%")
st.write(f"Average Confidence Percentage for Negative Feedback: {average_confidence_negative:.2f}%")


# footer

# line separator
st.markdown('<hr style="border: 2px solid #ddd;">', unsafe_allow_html=True)

# footer text
st.markdown(
    """
    <div style="text-align: center; padding: 10px;">
        Developed by <a href="https://github.com/Abdulraqib20" target="_blank">raqibcodes</a>
    </div>
    """,
    unsafe_allow_html=True
)
