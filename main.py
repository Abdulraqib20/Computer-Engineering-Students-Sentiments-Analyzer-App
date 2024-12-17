import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import sentiment_analysis as sa 
import yaml
import time
from src.viz.visuals import show_viz
from src.config.config import DATA_FILE, RELEVANT_COLUMNS, MODEL_NAME


# --- Configure Streamlit page ---
st.set_page_config(
    page_title="EduAI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'About': "Made with 💖 by raqibcodes"
    }
)


# --- Styling ---


# --- Header ---
st.markdown(
    """
    <div class="main-header">
        <h1>EduAI Analyzer<span>👨‍💻</span></h1>
        <p>Uncover insights from educational experiences of CPE students...</p>
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
            color: #fffff;
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
    """,
    unsafe_allow_html=True,
)


##-----------------------------------------------STYLE HEADER AND ABOUT SECTIONS--------------------------------------##

# Custom HTML, CSS, and JavaScript for animated tabs
custom_html = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');

.tabs-container {
    font-family: 'Poppins', sans-serif;
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    max-height: 500px; /* Adjust this value as needed */
    display: flex;
    flex-direction: column;
    height: 420px; /* Set a fixed height */
    display: flex;
    flex-direction: column;
}

.tab-buttons {
    display: flex;
    justify-content: center;
    margin-bottom: 20px;
}

.tab-button {
    background: none;
    border: none;
    padding: 10px 20px;
    font-size: 18px;
    font-weight: 500;
    color: #333;
    cursor: pointer;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.tab-button::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    width: 100%;
    height: 3px;
    background-color: #25D366;
    transform: scaleX(0);
    transition: transform 0.3s ease;
}

.tab-button:hover::after,
.tab-button.active::after {
    transform: scaleX(1);
}

.tab-content-container {
    flex: 1;
    overflow-y: auto;
    padding-right: 10px; /* Space for scrollbar */
}

.tab-content {
    background: rgba(255, 255, 255, 0.8);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transition: all 0.5s ease;
    opacity: 0;
    transform: translateY(20px);
    display: none;
    height: auto;
}

.tab-content.active {
    opacity: 1;
    transform: translateY(0);
    display: block;
}

.feature-list {
    list-style-type: none;
    padding: 0;
}

.feature-list li {
    margin-bottom: 10px;
    padding-left: 25px;
    position: relative;
}

.feature-list li::before {
    content: '🚀';
    position: absolute;
    left: 0;
    top: 0;
}

/* Customizing scrollbar */
.tab-content-container::-webkit-scrollbar {
    width: 8px;
}

.tab-content-container::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 10px;
}

.tab-content-container::-webkit-scrollbar-thumb {
    background: #888;
    border-radius: 10px;
}

.tab-content-container::-webkit-scrollbar-thumb:hover {
    background: #555;
}
</style>

<div class="tabs-container">
    <div class="tab-buttons">
        <button class="tab-button active" onclick="showTab('how-to-use')">How To Use</button>
        <button class="tab-button" onclick="showTab('about')">About</button>
    </div>
    
    <div class="tab-content-container">
        <div id="how-to-use" class="tab-content active">
        <h2>App Features</h2>
        <p>
        EduAI Analyzer is an educational experience analysis system developed by raqibcodes. 
        The system analyzes feedback text data from 300-level Undergraduate Computer Engineering Students 
        at the University of Ilorin using advanced Machine Learning algorithms and Generative AI technology.
        </p>
        <ul>
        <li><strong>Sentiment Analysis Functionality</strong>
        <ul>
        <li>Utilizes a sentiment analysis model to score feedback text.</li>
        <li>Analyzes sentiments as positive, neutral, or negative.</li>
        </ul>
        </li>
        <li><strong>User Input Collection</strong>
        <ul>
        <li>Gathers user's feedback and related information based on various criteria (course code, previous experience, gender, etc.).</li>
        </ul>
        </li>
        <li><strong>Text Preprocessing</strong>
        <ul>
        <li>Preprocesses the user feedback text using Natural Language Processing techniques.</li>
        </ul>
        </li>
        <li><strong>Percentage Confidence</strong>
        <ul>
        <li>Percentage level of how confident the model is making the prediction.</li>
        </ul>
        </li>
        <li><strong>Interactive Visualization</strong>
        <ul>
        <li>Provides various interactive plots for visualizing sentiment analysis results and other key metrics.</li>
        <li>Displays sentiment distribution in various charts (bar chart, pie chart, word cloud).</li>
        <li>Presents feedback counts based on course difficulty, course code, and gender.</li>
        <li>Provides insights into word frequency and usage in feedback.</li>
        <li>Explores the distribution of study hours, word count, and overall satisfaction.</li>
        </ul>
        </li>
        <li><strong>Summary Statistics</strong>
        <ul>
        <li>Offers a sentiment summary with counts of positive, neutral, and negative feedback including the percentage confidence results.</li>
        </ul>
        </li>
        <li><strong>Interactive Exploration</strong>
        <ul>
        <li>Allows users to trigger the exploration of visualizations by clicking a button.</li>
        </ul>
        </li>
        <li><strong>Real-Time Feedback Data Access</strong>
        <ul>
        <li>The app allows users access to real-time feedback data after getting their prediction results.</li>
        <li>Users can download and view the data directly within the app.</li>
        </ul>
        </li>
        <li><strong>Automatic Real-Time Saving</strong>
        <ul>
        <li>The app works in real-time, automatically saving prediction results and other insights generated.</li>
        </ul>
        </li>
        </ul>
    </div>

    
    <div id="about" class="tab-content">
    <h2>About</h2>
    <p>
        This web app is a sophisticated sentiment analysis tool developed by RAQIBCODES. It offers powerful features for text analysis and interaction:
    </p>

    <h3>✨ Key Features:</h3>
    <ul class="feature-list">
        <li>Detect Positive, Neutral, or Negative sentiments in user-entered text</li>
        <li>Trained on feedback from 300-level Undergraduate Computer Engineering Students at the University of Ilorin</li>
        <li>Utilizes fine-tuned BERT model and KerasNLP techniques with 96% accuracy</li>
        <li>Incorporates RoBERTa-based model for additional sentiment evaluation</li>
        <li>Includes Generative AI capabilities for interactive chats about the feedback data</li>
    </ul>

    <h3>✨ Objectives:</h3>
    <ul class="feature-list">
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
    
    </div>

    
</div>

<script>
function showTab(tabId) {
    // Hide all tab contents
    var tabContents = document.getElementsByClassName('tab-content');
    for (var i = 0; i < tabContents.length; i++) {
        tabContents[i].classList.remove('active');
    }
    
    // Show the selected tab content
    document.getElementById(tabId).classList.add('active');
    
    // Update active state of tab buttons
    var tabButtons = document.getElementsByClassName('tab-button');
    for (var i = 0; i < tabButtons.length; i++) {
        tabButtons[i].classList.remove('active');
    }
    event.currentTarget.classList.add('active');
}
</script>
"""

# Render the custom HTML
components.html(custom_html, height=600)

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

# # --- Tabs ---
# tab1, tab2 = st.tabs(["🎯 Objectives", "🎯 App Features"])

# # --- Content ---
# with tab1:

#     st.markdown(
#         """
#         <ul>
#             <li>Uncover and interpret sentiments in student feedback to understand their perceptions, satisfaction, and areas where improvements are needed.</li>
#             <li>Perform real-time sentiment analysis to provide immediate insights into current student opinions and feelings.</li>
#             <li>Present sentiment trends over time using interactive visualizations to highlight changes and patterns.</li>
#             <li>Extract valuable insights related to teaching methodologies, individual lecturers, and specific departmental courses.</li>
#             <li>Identify and emphasize specific challenges students face, enabling targeted interventions and improvements.</li>
#             <li>Create an interactive environment where users can explore and understand sentiment analysis results in depth.</li>
#             <li>Establish a continuous feedback loop between students and faculty to foster ongoing improvement in educational practices.</li>
#             <li>Allow lecturers to download sentiment analysis data for further, more detailed analysis outside the application.</li>
#             <li>Ensure the privacy and ethical handling of all student feedback data, adhering to relevant regulations.</li>
#             <li>Guide and support lecturers in interpreting and effectively utilizing sentiment analysis results to enhance their teaching.</li>
#         </ul>
#         """,
#         unsafe_allow_html=True,
#     )
    
# with tab2:
    
#     st.markdown(
#         """
#         1. **Sentiment Analysis Functionality**
#             - Utilizes a sentiment analysis model to score feedback text.
#             - Analyzes sentiments as positive, neutral, or negative.
#         <br>

#         2. **User Input Collection**
#             - Gathers user's feedback and related information based on various criteria (course code, previous experience, gender, etc.).
#         <br>
#         3. **Text Preprocessing**
#             - Preprocesses the user feedback text using Natural Language Processing techniques.
#         <br>
#         4. **Percentage Confidence**
#             - Percentage level of how confident the model is making the prediction.
#         <br>
#         5. **Interactive Visualization**
#             - Provides various interactive plots for visualizing sentiment analysis results and other key metrics.
#             - Displays sentiment distribution in various charts (bar chart, pie chart, word cloud).
#             - Presents feedback counts based on course difficulty, course code, and gender.
#             - Provides insights into word frequency and usage in feedback.
#             - Explores the distribution of study hours, word count, and overall satisfaction.
#         <br>
#         6. **Summary Statistics**
#             - Offers a sentiment summary with counts of positive, neutral, and negative feedback including the percentage confidence results.
#         <br>
#         8. **Interactive Exploration**
#             - Allows users to trigger the exploration of visualizations by clicking a button.
#         <br>
#         9. **Real-Time Feedback Data Access**
#             - The app allows users access to real-time feedback data after getting their prediction results.
#             - Users can download and view the data directly within the app.
#         <br>
#         10.  **Automatic Real-Time Saving**
#             - The app works in real-time, automatically saving prediction results and other insights generated.
#         """,
#         unsafe_allow_html=True,
#     )




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


@st.cache_data(show_spinner=False)
def load_data(filename):
    df = pd.read_csv(filename)
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')
    df['processed_feedback'] = df['feedback'].astype(str).apply(sa.preprocess_text)
    df['char_count'] = df['processed_feedback'].apply(len)
    df['word_count'] = df['processed_feedback'].apply(lambda x: len(x.split()))
    return df

df = load_data(DATA_FILE)

# Load data
try:
    df = load_data(DATA_FILE)
    st.write("Data loaded successfully from CSV file!")
except Exception as e:
    st.error("Unable to load data from database!")



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
           
            # --- Save updated data ---
            try:
                df.to_csv(DATA_FILE, index=False)  
                st.success("Sentiment analysis completed!")
            except PermissionError:
                st.error("Error saving data: Please close the 'survey_data.csv' file if it's open and try again.")
            except Exception as e:  # Catch other potential errors
                st.error(f"Error saving data: {str(e)}")
            
            # --- Download button ---
            st.download_button(
                label="Download Updated Dataset",
                data=df.to_csv(index=False).encode(),
                file_name='survey_data_updated.csv',
                mime='text/csv'
            )



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

###----------------------------------------CHATTING & VISUALIZATIONS--------------------------------------------------------------####

    #----------------------------------------------STYLING TABS---------------------------------------------------------#

# Title for the tabs section
st.markdown("<h2 style='text-align: center; margin-bottom: 10px;'>Additional Features</h2>", 
                unsafe_allow_html=True)

    # Tabs for different sections of the app
sec1, sec2 = st.tabs(["📊 Visualizations", "💬 ChatGPT"])
    
#-----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------VIZUALIZATION SECTION-----------------------------------------------------#
    
    
with sec1:
        
    # st.markdown("""
    #     <style>
    #         .stTabs {
    #             overflow-x: auto;
    #         }
    #         .stTabs [data-baseweb="tab-list"] {
    #             display: flex !important;
    #             flex-wrap: nowrap !important;
    #             overflow-x: auto !important;
    #             white-space: nowrap !important;
    #             border-bottom: none !important;
    #             -webkit-overflow-scrolling: touch !important;
    #             background-color: #075E54 !important; /* WhatsApp dark green */
    #         }
    #         .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
    #             display: none !important;
    #         }
    #         .stTabs [data-baseweb="tab-list"] {
    #             -ms-overflow-style: none !important;
    #             scrollbar-width: none !important;
    #         }
    #         .stTabs [data-baseweb="tab"] {
    #             flex: 0 0 auto !important;
    #             padding: 10px 20px !important;
    #             font-size: 16px !important;
    #             cursor: pointer !important;
    #             background-color: #075E54 !important; /* WhatsApp dark green */
    #             color: #ffffff !important;
    #             border: none !important;
    #             transition: background-color 0.3s ease, color 0.3s ease !important;
    #             margin-right: 5px !important;
    #         }
    #         .stTabs [data-baseweb="tab"]:hover {
    #             background-color: #128C7E !important; /* WhatsApp light green */
    #         }
    #         .stTabs [aria-selected="true"] {
    #             color: #075E54 !important; /* WhatsApp dark green */
    #             background-color: #ffffff !important;
    #             border-top-left-radius: 5px !important;
    #             border-top-right-radius: 5px !important;
    #         }
    #         .stTabs [data-baseweb="tab-panel"] {
    #             padding: 20px !important;
    #         }
    #     </style>
    #     """, unsafe_allow_html=True)   

    
    show_viz(df)

#----------------------------------------------CHATGPT SECTION----------------------------------------------------------#
# with sec2:

#     from app import main
#     main()

#     if __name__ == "__main__":
#         main()
    
    
    
  
#-------------------------------------------------------------FOOTER----------------------------------------------------#

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Raleway:wght@300;400;600&display=swap');

        .footer-container {
            font-family: 'Raleway', sans-serif;
            margin-top: 50px;
            padding: 30px 0;
            width: 100vw;
            position: absolute;
            left: 50%;
            right: 50%;
            margin-left: -50vw;
            margin-right: -50vw;
            # overflow: hidden;
        }

        .footer-content {
            display: flex;
            justify-content: center;
            align-items: center;
            position: relative;
            z-index: 2;
        }

        .footer-text {
            color: #ffffff;
            font-size: 20px;
            font-weight: 300;
            text-align: center;
            margin: 0;
            padding: 0 20px;
            position: relative;
        }

        .footer-link {
            color: #075E54;  /* WhatsApp dark green */
            font-weight: 600;
            text-decoration: none;
            position: relative;
            transition: all 0.3s ease;
            padding: 5px 10px;
            border-radius: 5px;
        }

        .footer-link:hover {
            background-color: rgba(7, 94, 84, 0.1);  /* Slightly darker on hover */
            box-shadow: 0 0 15px rgba(7, 94, 84, 0.2);
        }

        .footer-heart {
            display: inline-block;
            color: #FF0000;  /* Red heart */
            font-size: 35px;
            animation: pulse 1.5s ease infinite;
        }

        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.1); }
            100% { transform: scale(1); }
        }
    </style>

    <div class="footer-container">
        <div class="footer-content">
            <p class="footer-text">
                Made with <span class="footer-heart">♥</span> by 
                <a href="https://github.com/Abdulraqib20" target="_blank" class="footer-link">raqibcodes</a>
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

