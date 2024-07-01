import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import sentiment_analysis as sa 
import yaml
from sqlalchemy import create_engine, inspect
from sqlalchemy.types import Integer, Text, Date, Time, Float

# --- Configure Streamlit page ---
st.set_page_config(
    page_title="CPE APP",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'About': "Made with 💖 by raqibcodes"
    }
)

# Load configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

DATA_FILE = config['data_file']
RELEVANT_COLUMNS = config['relevant_columns']
DB_URI = config['db_uri']

# --- Styling ---


# --- Header ---
st.markdown(
    """
    <div class="main-header">
        <h1>Sentiment Analyzer<span>🤖</span></h1>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
        .main-header {
            text-align: center;
            border-radius: 15px; /* More rounded corners for a softer look */
            box-shadow: 4px 4px 15px rgba(0,0,0,0.1); /* Enhanced shadow for more depth */
            font-family: 'Arial', sans-serif; /* Custom font family for a modern look */
            margin: 20px 0; /* Margin to create space around the header */
        }

        .main-header h1 {
            font-size: 3rem; /* Larger font size for the header */
            margin: 0; /* Remove default margin */
            animation: fadeIn 2s ease-in-out; /* Add fade-in animation */
        }

        /* Style for the emoji */
        .main-header h1 span {
            display: inline-block;
            animation: bounce 2s infinite; /* Add bounce animation */
        }

        @keyframes bounce {
            0%, 20%, 50%, 80%, 100% {
                transform: translateY(0);
            }
            40% {
                transform: translateY(-20px);
            }
            60% {
                transform: translateY(-10px);
            }
        }

        @keyframes fadeIn {
            from {
                opacity: 0;
            }
            to {
                opacity: 1;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

## title
st.markdown(
    """
    <style>
        .title-container {
            text-align: center;
            color: #ffffe;
            font-family: 'Montserrat', sans-serif;
            transition: transform 0.2s ease; 
        }
        .title-container:hover {
            transform: translateY(-3px);
        }
        .title {
            font-size: 2rem; 
            margin-bottom: 5px; 
        }
        .subtitle {
            font-size: 1.2rem; 
            font-weight: 300;
            color: #aaa; /* Slightly darker gray for subtitle */
        }
        .emoji {
            font-size: 2.5rem; 
            margin-bottom: 10px; 
        }
    </style>
    <div class="title-container">
        <span class="emoji">🧠</span> <span class="title">Emotion Detection</span>
        <div class="subtitle">Unleash Your Emotional Intelligence</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.title('')

st.markdown(
    """
    <style>
        .intro-section, .get-started-section {
            # background-color: #fff; /* White background */
            border: 1px solid #ddd; /* Subtle border */
            padding: 25px; /* More padding for better readability */
            border-radius: 10px; /* Rounded corners for a softer look */
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1); /* Subtle shadow for depth */
        }

        .intro-section h2, .get-started-section h3 {
            color: #fffff;
            margin-bottom: 15px;
        }

        .intro-section p, .get-started-section p {
            line-height: 1.6;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Introduction ---
with st.container():
    with st.expander("About the App"):
        st.markdown(
            """
            <div class="intro-section">
                <p>
                    This web app is a sentiment analysis tool developed by raqibcodes. It has the capability of detecting whether user-entered text has an underlying Positive, 
                    Neutral or Negative sentiment. The text classification model was trained on Feedback survey data collected from 300-level Undergraduate Computer Engineering 
                    Students at the University of Ilorin (who are His's peers). The model underwent fine-tuning using the a BERT model and KerasNLP
                    techniques, resulting in an impressive accuracy score of 96%. The data was subsequently evaluated using a RoBERTa-based model
                    which is a transformer-based model and it also showed strong performances in analyzing sentiments accurately.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

#---Get Started ---

with st.container():
    st.markdown(
    """
    <div class="get-started-section">
      <h3>Get Started</h3>
      <p>
        Just complete all the fields and type in your message, and it will quickly show you the underlying emotion and the percentage level of confidence.
      </p>
    </div>
    """,
    unsafe_allow_html=True
    )

    
st.title(" ")
st.title(" ")

# --- Styling ---
st.markdown(
    """
    <style>
        .stTabs [data-baseweb="tab-list"] {
            display: flex;
            justify-content: center;
        }

        .stTabs [data-baseweb="tab"] {
            color: #E4E4E4; /* Default tab text color */
            padding: 15px 30px;
            border: none;
            border-radius: 8px;
            margin: 0 5px;
            cursor: pointer;
            font-weight: bold;
            transition: color 0.3s ease, transform 0.3s ease, box-shadow 0.3s ease;
        }

        .stTabs [data-baseweb="tab"]:hover {
            color: #D13CC1; /* Hover color change */
            transform: translateY(-2px); /* Lift on hover */
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); /* Subtle shadow on hover */
        }

        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            color: #D13CC1; /* Active tab text color */
            border-bottom: 2px solid #D13CC1; /* Underline for active tab */
        }

        .stTabs [data-baseweb="tab-panel"] {
            padding: 0; /* Remove default padding */
        }

        .stExpanderHeader { /* Style the expander header */
            background-color: #f5f5f5;
            padding: 15px;
            border-radius: 8px;
            margin-top: 10px; /* Add some spacing */
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Tabs ---
tab1, tab2 = st.tabs(["🎯 Objectives", "🎯 App Features"])

# --- Content ---
with tab1:
    with st.expander(" "):
        st.markdown(
            """
            <ul>
                <li>Uncover and interpret sentiments in student feedback to understand their perceptions, satisfaction, and areas where improvements are needed.</li>
                <li>Perform real-time sentiment analysis to provide immediate insights into current student opinions and feelings.</li>
                <li>Present sentiment trends over time using interactive visualizations to highlight changes and patterns.</li>
                <li>Extract valuable insights related to teaching methodologies, individual lecturers, and specific departmental courses.</li>
                <li>Identify and emphasize specific challenges students face, enabling targeted interventions and improvements.</li>
                <li>Create an interactive environment where users can explore and understand sentiment analysis results in depth.</li>
                <li>Establish a continuous feedback loop between students and faculty to foster ongoing improvement in educational practices.</li>
                <li>Allow lecturers to download sentiment analysis data for further, more detailed analysis outside the application.</li>
                <li>Ensure the privacy and ethical handling of all student feedback data, adhering to relevant regulations.</li>
                <li>Guide and support lecturers in interpreting and effectively utilizing sentiment analysis results to enhance their teaching.</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )
    
with tab2:
    with st.expander(" "):
        st.markdown(
            """
            1. **Sentiment Analysis Functionality**
                - Utilizes a sentiment analysis model to score feedback text.
                - Analyzes sentiments as positive, neutral, or negative.
            <br>

            2. **User Input Collection**
                - Gathers user's feedback and related information based on various criteria (course code, previous experience, gender, etc.).
            <br>
            3. **Text Preprocessing**
                - Preprocesses the user feedback text using Natural Language Processing techniques.
            <br>
            4. **Percentage Confidence**
               - Percentage level of how confident the model is making the prediction.
            <br>
            5. **Interactive Visualization**
               - Provides various interactive plots for visualizing sentiment analysis results and other key metrics.
               - Displays sentiment distribution in various charts (bar chart, pie chart, word cloud).
               - Presents feedback counts based on course difficulty, course code, and gender.
               - Provides insights into word frequency and usage in feedback.
               - Explores the distribution of study hours, word count, and overall satisfaction.
            <br>
            6. **Summary Statistics**
               - Offers a sentiment summary with counts of positive, neutral, and negative feedback including the percentage confidence results.
            <br>
            8. **Interactive Exploration**
               - Allows users to trigger the exploration of visualizations by clicking a button.
            <br>
            9. **Real-Time Feedback Data Access**
                - The app allows users access to real-time feedback data after getting their prediction results.
                - Users can download and view the data directly within the app.
            <br>
            10.  **Automatic Real-Time Saving**
                - The app works in real-time, automatically saving prediction results and other insights generated.
            """,
            unsafe_allow_html=True,
        )




# --- Styling ---
st.markdown(
    """
    <style>
        .selectbox-container {
            background-color: #f5f5f5; /* Light gray background */
            border: 1px solid #ddd; /* Subtle border */
            padding: 20px;
            border-radius: 10px; /* Rounded corners */
            margin-bottom: 10px; /* Add spacing between containers */
        }

        .selectbox-container label {
            font-weight: bold;
            color: #333; /* Darker label color */
        }

        .stSelectbox {
            width: 100%; /* Make selectboxes take full width */
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Input Section ---

# Initialize Session State Variables
for key in RELEVANT_COLUMNS:
    if key not in st.session_state:
        st.session_state[key] = 'Select Option' if key != 'study_hours (per week)' else 0

st.markdown("<h3 style='text-align: center;'>Student Details</h3>", unsafe_allow_html=True)

# Define callback functions for each selectbox
def update_course_code():
    st.session_state.course_code = st.session_state["course_code"]

def update_difficulty():
    st.session_state.difficulty = st.session_state["difficulty"]

def update_previous_exp():
    st.session_state.previous_exp = st.session_state["previous_exp"]

def update_gender():
    st.session_state.gender = st.session_state["gender"]

def update_department():
    st.session_state.department = st.session_state["department"]

def update_attendance():
    st.session_state.attendance = st.session_state["attendance"]

def update_study_hours():
    st.session_state.study_hours = st.session_state["study_hours (per week)"]

def update_satisfaction():
    st.session_state.satisfaction = st.session_state["satisfaction"]

# Streamlit form to capture user input

with st.container():
    col1, col2, col3 = st.columns(3)

    with col1:
        with st.expander("Course Information"):
            st.session_state['course_code'] = st.selectbox("Course Code", 
                                                           ['Select Course Code', 'CPE 321', 
                                                            'CPE 311', 'CPE 341', 'CPE 381', 
                                                            'CPE 331', 'MEE 361', 'GSE 301'])
            
            st.session_state['difficulty'] = st.selectbox("Course Difficulty", 
                                                          ['Select Difficulty', 'Easy', 'Difficult', 
                                                           'Challenging', 'Moderate'])


    with col2:
        with st.expander("Student Demographics"):
            st.session_state['previous_experience'] = st.selectbox("Previous Experience", [
                'Select Option', "Yes", "No"])
            
            st.session_state['gender'] = st.selectbox("Gender", 
                                                      ['Select Gender', 'Male', 'Female'])
            
            st.session_state['department'] = st.selectbox("Department", 
                                                          ['Select Option', "Yes", "No"]) 
            
            
    with col3:
        with st.expander("Additional Information"):
            st.session_state['attendance'] = st.selectbox("Attendance", 
                                                          ['Select Attendance', 'Regular', 
                                                           'Irregular', 'Occasional'])
            
            st.session_state['study_hours (per week)'] = st.selectbox("Study Hours (per week)", 
                                                                      options=['Select Study Hours'] + list(range(25)))
            
            st.session_state['overall_satisfaction'] = st.selectbox("Overall Satisfaction", 
                                                                    options=['Select Overall Satisfaction']
                                                                    + list(range(1, 11)))


# Database connection
engine = create_engine(DB_URI)
# Check if the table exists
inspector = inspect(engine)
table_exists = inspector.has_table('feedback')

# Load data from CSV and save to database
@st.cache_resource
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')
    df['processed_feedback'] = df['feedback'].astype(str).apply(sa.preprocess_text)
    df['char_count'] = df['processed_feedback'].apply(len)
    df['word_count'] = df['processed_feedback'].apply(lambda x: len(x.split()))
    df.to_sql('feedback', engine, if_exists='replace', index=False, dtype={
        'course_code': Text,
        'feedback': Text,
        'previous_experience': Text,
        'gender': Text,
        'attendance': Text,
        'course_difficulty': Text,
        'study_hours (per week)': Integer,
        'overall_satisfaction': Integer,
        'department': Text,
        'date': Date,
        'time': Time,
        'hour': Integer,
        'processed_feedback': Text,
        'char_count': Integer,
        'word_count': Integer,
        'sentiments': Text,
        'sentiments_index': Integer,
        'percentage_confidence': Text
    })
    return df

# Load data from database
@st.cache_resource
def load_data_from_db():
    query = "SELECT * FROM feedback"
    df = pd.read_sql(query, engine)
    return df

# Save results to database
def save_to_db(df):
    df.to_sql('feedback', engine, if_exists='replace', index=False, dtype={
        'course_code': Text,
        'feedback': Text,
        'previous_experience': Text,
        'gender': Text,
        'attendance': Text,
        'course_difficulty': Text,
        'study_hours (per week)': Integer,
        'overall_satisfaction': Integer,
        'department': Text,
        'date': Date,
        'time': Time,
        'hour': Integer,
        'processed_feedback': Text,
        'char_count': Integer,
        'word_count': Integer,
        'sentiments': Text,
        'sentiments_index': Integer,
        'percentage_confidence': Text
    })

# @st.cache_data(show_spinner=False)
# def load_data(filename):
#     df = pd.read_csv(filename)
#     df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')
#     df['processed_feedback'] = df['feedback'].astype(str).apply(sa.preprocess_text)
#     df['char_count'] = df['processed_feedback'].apply(len)
#     df['word_count'] = df['processed_feedback'].apply(lambda x: len(x.split()))
#     return df

# df = load_data(DATA_FILE)

# Load data
try:
    df = load_data_from_db()
    st.write("Data loaded successfully from the database!")
except Exception as e:
    st.write("No data in database. Loading from CSV file...")
    df = load_data(DATA_FILE)
    st.write("Data loaded successfully from CSV and saved to the database!")



user_input = ""
user_input = st.text_area("Enter Your Text Feedback Here:", value=user_input)
if st.button("Analyze Sentiment"):
     # Define the placeholder options for the relevant columns
    # placeholders = {
    #     'course_code': 'Select Course Code',
    #     'previous_experience': 'Select Option',
    #     'gender': 'Select Gender',
    #     'attendance': 'Select Attendance',
    #     'difficulty': 'Select Difficulty',
    #     'study_hours (per week)': 'Select Study Hours',
    #     'satisfaction': 'Select Overall Satisfaction',
    #     'department': 'Select Option'
    # }
    if not user_input.strip():  
        st.warning("Please enter your feedback.")
    # Check if all fields are completed
    # elif any(st.session_state.get(key, placeholders[key]) == placeholders[key] for key in RELEVANT_COLUMNS):
    #     st.warning("Please complete all the fields.")
    else:
        with st.spinner("Analyzing sentiment..."):  # Progress indicator
            # Retrieve values from the sideboxes only when there is user input
            course_code = st.session_state.course_code or 'Select Course Code'
            previous_experience = st.session_state.previous_experience
            gender = st.session_state.gender
            attendance = st.session_state.attendance
            difficulty = st.session_state.difficulty
            study_hours = st.session_state['study_hours (per week)']
            satisfaction = st.session_state.overall_satisfaction
            department = st.session_state.department
            
            user_input_processed = sa.preprocess_text(user_input)
            result = sa.sentiment_score(user_input_processed)
            
                        # --- Display Result ---
            st.subheader("Sentiment Analysis Result:")
            st.write(f"Predicted Sentiment: {result.get('predicted_label', 'N/A')}")
            st.write(f"Percentage Confidence of Prediction: {result.get('confidence_percentage', 'N/A')}")
            # Display interpreted emotions
            emotion_interpretation = sa.interpret_emotions(result.get('emotions', {}))
            st.write(f"Emotions Expressed: {emotion_interpretation}")
            
            # Display explanation based on keywords and emotions
            explanation = sa.get_explanation(user_input_processed, result)
            st.write("Explanation:", explanation)
            
            # --- Update DataFrame (use st.session_state here) ---
            new_feedback = pd.DataFrame({
                # **{col: [st.session_state.get(col, 'Select Option')] for col in RELEVANT_COLUMNS}, 
                'course_code': [course_code], 
                'feedback': [user_input],
                'previous_experience': [previous_experience],
                'gender': [gender],
                'attendance': [attendance],
                'course_difficulty': [difficulty],
                'study_hours (per week)': [study_hours],
                'overall_satisfaction': [satisfaction],
                'department': [department],
                'date': [pd.to_datetime('now').date()],
                'time': [pd.to_datetime('now').time()],
                'hour': [pd.to_datetime('now').hour],
                'processed_feedback': [user_input_processed],
                'char_count': [len(user_input_processed)],
                'word_count': [len(user_input_processed.split())],
            })
            # Apply sentiment scoring to ONLY the new_feedback DataFrame
            new_feedback = sa.apply_sentiment_scoring(new_feedback)
            df = pd.concat([df, new_feedback], ignore_index=True)  # Add to DataFrame
           
            # # --- Save updated data ---
            # try:
            #     df.to_csv(DATA_FILE, index=False)  
            #     st.success("Sentiment analysis completed!")
            # except PermissionError:
            #     st.error("Error saving data: Please close the 'survey_data.csv' file if it's open and try again.")
            # except Exception as e:  # Catch other potential errors
            #     st.error(f"Error saving data: {str(e)}")
            
            # --- Save updated data to the database ---
            try:
                save_to_db(df)
                st.success("Sentiment analysis completed and data saved to the database!")
            except Exception as e:
                st.error(f"Error saving data: {str(e)}")


            # # --- Download button ---
            # st.download_button(
            #     label="Download Updated Dataset",
            #     data=df.to_csv(index=False).encode(),
            #     file_name='survey_data_updated.csv',
            #     mime='text/csv'
            # )
            
            # --- Download button ---
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

        label_data = df['sentiments'].value_counts()
        fig = px.pie(label_data, values=label_data.values, names=label_data.index, hole=0.4)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(title="Sentiments Distribution (Pie Chart)")
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

# df['percentage_confidence'] = df['percentage_confidence'].str.rstrip('%').astype(float)
# df['percentage_confidence'] = df['percentage_confidence'].apply(lambda x: f"{x}%")
# df['percentage_confidence'] = df['percentage_confidence'].apply(lambda x: float(x.strip('%')))
df['percentage_confidence'] = df['percentage_confidence'].astype(str).str.rstrip('%').astype(float)

# --- Sentiment Summary ---
positive_feedback_count = (df["sentiments_index"] == 3).sum()
neutral_feedback_count = (df["sentiments_index"] == 2).sum()
negative_feedback_count = (df["sentiments_index"] == 1).sum()

# Calculate average confidence percentage for each sentiment
average_confidence_positive = df.loc[df["sentiments_index"] == 3, "percentage_confidence"].mean()
average_confidence_neutral = df.loc[df["sentiments_index"] == 2, "percentage_confidence"].mean()
average_confidence_negative = df.loc[df["sentiments_index"] == 1, "percentage_confidence"].mean()

st.markdown("<h3 style='text-align: center;'>Sentiment Summary</h3>", unsafe_allow_html=True)

# Create columns for layout
summary_col1, summary_col2, summary_col3 = st.columns(3)

# Display positive feedback summary
with summary_col1:
    st.markdown(
        f"""
        <div class="summary-card">
            <h4 style='text-align: center;'>Positive Feedback</h4>
            <p style='text-align: center; font-size: 24px;'>{positive_feedback_count}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Display neutral feedback summary
with summary_col2:
    st.markdown(
        f"""
        <div class="summary-card">
            <h4 style='text-align: center;'>Neutral Feedback</h4>
            <p style='text-align: center; font-size: 24px;'>{neutral_feedback_count}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Display negative feedback summary
with summary_col3:
    st.markdown(
        f"""
        <div class="summary-card">
            <h4 style='text-align: center;'>Negative Feedback</h4>
            <p style='text-align: center; font-size: 24px;'>{negative_feedback_count}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- CSS Styling ---
st.markdown(
    """
    <style>
        /* ... your existing styles ... */

        .summary-card {
            # background-color: #fff;
            border: 1px solid #ddd;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---Footer---

# footer text

st.markdown(
    """
    <style>
        div.stMarkdown footer {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 25px;
            background: linear-gradient(to right, #ff69b4, #9400d3); /* Vibrant pink/purple */
            color: white;
            font-size: 18px;
            border-radius: 15px;
            margin-top: 40px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5); /* Subtle text shadow for depth */
        }

        div.stMarkdown footer p { margin: 0; } 

        div.stMarkdown footer a {
            color: inherit;
            font-weight: bold;
            position: relative;
            transition: all 0.3s ease;
            text-shadow: none;
        }

        div.stMarkdown footer a::before { /* Glowing underline effect */
            content: "";
            position: absolute;
            width: 100%;
            height: 2px;
            bottom: -5px;
            left: 0;
            background-color: #fff;
            visibility: hidden;
            transform: scaleX(0);
            transition: all 0.3s ease-in-out 0s;
        }

        div.stMarkdown footer a:hover::before {
            visibility: visible;
            transform: scaleX(1);
        }

        div.stMarkdown footer a:hover {
            color: #f0f0f0; /* Slightly lighter white on hover */
            letter-spacing: 1px;
        }
    </style>
    <footer>
        <p>
            Made with ❤️ by&nbsp;
            <a href="https://github.com/Abdulraqib20" target="_blank">raqibcodes</a>
        </p>
    </footer>
    """,
    unsafe_allow_html=True,
)
