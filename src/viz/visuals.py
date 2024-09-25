# Import libraries and packages
import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
import collections
from wordcloud import WordCloud
import warnings;warnings.filterwarnings(action='ignore')
#-----------------------------------------------STYLING TABS-------------------------------------------------------------#
def show_viz(df):
    st.markdown("""
        <style>
            .stTabs {
                overflow-x: auto;
            }
            .stTabs [data-baseweb="tab-list"] {
                display: flex !important;
                flex-wrap: nowrap !important;
                overflow-x: auto !important;
                white-space: nowrap !important;
                border-bottom: none !important;
                -webkit-overflow-scrolling: touch !important;
                background-color: #075E54 !important; /* WhatsApp dark green */
            }
            .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
                display: none !important;
            }
            .stTabs [data-baseweb="tab-list"] {
                -ms-overflow-style: none !important;
                scrollbar-width: none !important;
            }
            .stTabs [data-baseweb="tab"] {
                flex: 0 0 auto !important;
                padding: 10px 20px !important;
                font-size: 16px !important;
                cursor: pointer !important;
                background-color: #075E54 !important; /* WhatsApp dark green */
                color: #ffffff !important;
                border: none !important;
                transition: background-color 0.3s ease, color 0.3s ease !important;
                margin-right: 5px !important;
            }
            .stTabs [data-baseweb="tab"]:hover {
                background-color: #128C7E !important; /* WhatsApp light green */
            }
            .stTabs [aria-selected="true"] {
                color: #075E54 !important; /* WhatsApp dark green */
                background-color: #ffffff !important;
                border-top-left-radius: 5px !important;
                border-top-right-radius: 5px !important;
            }
            .stTabs [data-baseweb="tab-panel"] {
                padding: 20px !important;
            }
        </style>
        """, unsafe_allow_html=True)   

    st.header('Visualizations & Charts')

    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab8, tab9, tab10, tab11, tab12, tab13, tab14 = st.tabs([
        "Sentiment Overview",
        "Course Complexity",
        "Course Popularity",
        "Gender Insights",
        "Key Feedback Terms",
        "Course Sentiment Analysis",
        "Difficulty vs. Sentiment",
        "Gender Sentiment Trends",
        "Study Time & Satisfaction",
        "Study Habits Impact",
        "Feedback Length by Difficulty",
        "Satisfaction-Sentiment Correlation",
        "Sentiment Timeline"
        ])

#------------------------------------------------------------------------------------------------------------------------#
    with tab1:
        st.write("Content for Sentiments Distribution")
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
        
        
#------------------------------------------------------------------------------------------------------------------------#
    with tab2:
        course_difficulty_counts = df['course_difficulty'].value_counts()
        fig = px.bar(course_difficulty_counts, x=course_difficulty_counts.index, y=course_difficulty_counts.values, labels={'x': 'Course Difficulty', 'y': 'Count'})
        fig.update_layout(
            xaxis=dict(type='category'),
            title="Feedback Count by Course Difficulty",
            xaxis_title="Course Difficulty",
            yaxis_title="Count",
        )
        st.plotly_chart(fig)

#------------------------------------------------------------------------------------------------------------------------#
    with tab3:
        course_code_counts = df['course_code'].value_counts()
        fig = px.bar(course_code_counts, x=course_code_counts.index, y=course_code_counts.values, labels={'x': 'Course Code', 'y': 'Count'})
        fig.update_layout(
            xaxis=dict(type='category'),
            title="Feedback Count by Course Code",
            xaxis_title="Course Code",
            yaxis_title="Count",
        )
        st.plotly_chart(fig)
        
#------------------------------------------------------------------------------------------------------------------------#
    with tab4:
        gender_counts = df['gender'].value_counts()
        fig = px.pie(gender_counts, values=gender_counts.values, names=gender_counts.index, hole=0.4)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(title="Gender Distribution")
        st.plotly_chart(fig)
        
        
#------------------------------------------------------------------------------------------------------------------------#
    with tab5:
        from collections import Counter
        word_frequency = Counter(" ".join(df['feedback']).split()).most_common(30)
        word_df = pd.DataFrame(word_frequency, columns=['Word', 'Frequency'])
        fig = px.bar(word_df, x='Frequency', y='Word', orientation='h', title='Top 30 Most Frequently Used Words')
        st.plotly_chart(fig)
    
#------------------------------------------------------------------------------------------------------------------------#
    with tab6:
        fig = px.histogram(df, x='course_code', color='sentiments', title='Course Code distribution by Sentiment distribution')
        fig.update_xaxes(title='Course Code')
        fig.update_yaxes(title='Count of Feedback')
        st.plotly_chart(fig)

#------------------------------------------------------------------------------------------------------------------------#
    with tab8:
        fig = px.histogram(df, x='course_difficulty', color='sentiments', 
                           title='Sentiment Distribution by Course Difficulty',
                           category_orders={"Course Difficulty": ['Easy', 'Moderate', 'Challenging', 'Difficult']})
        fig.update_xaxes(title='Course Difficulty')
        fig.update_yaxes(title='Count of Feedback')
        st.plotly_chart(fig)
    
    
#------------------------------------------------------------------------------------------------------------------------#
    with tab9:
        fig = px.histogram(df, x='gender', color='sentiments', title='Sentiment Distribution by Gender')
        fig.update_xaxes(title='Gender')
        fig.update_yaxes(title='Count of Feedback')
        st.plotly_chart(fig)


#------------------------------------------------------------------------------------------------------------------------#
    with tab10:
        fig = px.scatter(df, x='study_hours (per week)', y='overall_satisfaction')
        fig.update_layout(
            title="Distribution of Study Hours (per week) and Overall Satisfaction",
            xaxis_title="Study Hours (per week)",
            yaxis_title="Overall Satisfaction",
        )
        st.plotly_chart(fig)

#------------------------------------------------------------------------------------------------------------------------#
    with tab11:
        fig = px.histogram(df, x='study_hours (per week)', color='sentiments', 
                           title='Distribution of Study Hours by Sentiment')
        fig.update_xaxes(title='Study Hours per Week')
        fig.update_yaxes(title='Count of Feedback')
        st.plotly_chart(fig)

#------------------------------------------------------------------------------------------------------------------------#
    with tab12:
        fig = px.box(df, x='course_difficulty', y='word_count', 
                         title='Distribution of Word Count for different levels of Course Difficulty',
                         category_orders={"course difficulty": ['Easy', 'Moderate', 'Challenging', 'Difficult']})
        fig.update_xaxes(title='Course Difficulty')
        fig.update_yaxes(title='Word Count')
        st.plotly_chart(fig)

#------------------------------------------------------------------------------------------------------------------------#
    with tab13:
        fig = px.box(df, x='sentiments', y='overall_satisfaction', 
                         title='Overall Satisfaction vs. Sentiment')
        fig.update_xaxes(title='Sentiment')
        fig.update_yaxes(title='Overall Satisfaction')
        st.plotly_chart(fig)
        
#------------------------------------------------------------------------------------------------------------------------#
    with tab14:
        fig = px.line(df, x='date', y='sentiments_index', 
                          title='Sentiment Over Time')
        fig.update_xaxes(title='Date')
        fig.update_yaxes(title='Sentiment Index')
        st.plotly_chart(fig)
