##--------------------Specifies how EduAI should generate queries for CSV data analysis.---------------------##
def get_columns_descriptions():
    COLUMNS_DESCRIPTIONS = {
        "course_code": {
            "description": "Unique identifier for each course",
            "examples": {
                "CPE 321": "Analogue Electronics",
                "GSE 301": "Entrepreneurship Skills",
                "CPE 331": "Electromagnetics",
                "CPE 381": "Laboratory Course",
                "CPE 311": "Electronics Course",
                "CPE 341": "Software Engineering Principles Course",
                "MEE 361": "Engineering Mathematics"
            }
        },
        "feedback": "User's textual feedback about the course or educational experience",
        "previous_experience": {
            "description": "Indicates if the user has prior experience with the course",
            "values": ["Yes", "No"]
        },
        "gender": {
            "description": "User's gender",
            "values": ["Male", "Female"]
        },
        "attendance": {
            "description": "User's attendance level",
            "values": ["Occasional", "Regular", "Irregular"]
        },
        "course_difficulty": {
            "description": "User's perception of course difficulty",
            "values": ["Challenging", "Moderate", "Easy", "Difficult"]
        },
        "study_hours": {
            "description": "Average weekly study hours for the course",
            "range": "1-24 hours"
        },
        "overall_satisfaction": {
            "description": "User's overall satisfaction with the course",
            "range": "Scale of 1-10"
        },
        "department": {
            "description": "Indicates if the user is a department member",
            "values": ["Yes", "No"]
        },
        "date": {
            "description": "Feedback submission date",
            "format": "YYYY-MM-DD"
        },
        "time": {
            "description": "Feedback submission time",
            "format": "HH:MM:SS"
        },
        "processed_feedback": "Pre-processed version of the feedback text (e.g., stop words removed, stemmed/lemmatized)",
        "char_count": "Total character count in the feedback text",
        "word_count": "Total word count in the feedback text",
        "sentiments": {
            "description": "Sentiment analysis result of the feedback",
            "values": ["Positive", "Negative", "Neutral"]
        },
        "sentiments_index": {
            "description": "Numerical representation of sentiments",
            "mapping": {
                "Positive": 3,
                "Negative": 1,
                "Neutral": 2
            }
        },
        "percentage_confidence": {
            "description": "Confidence score of sentiment analysis",
            "format": "Percentage"
        }
    }
    return COLUMNS_DESCRIPTIONS



###------------------------------------EduAI Query Generator-------------------------------------------------###
eduai_query_gen = """
You are a CSV file query generator. Your task is to generate queries based on user requests to analyze data stored in CSV files related to student feedback, courses, attendance, and other academic parameters. Follow these rules strictly:

This is what you do as a CSV query generator:
Given the columns below, you will be asked a question, and your job is to generate a logical query or analysis statement to address the user’s request. All queries should involve the appropriate fields such as course_code, attendance, or feedback.\

NOTE: 
1. For satisfaction analysis, consider the overall_satisfaction column. To filter based on feedback sentiment, use the sentiments_index column to focus on specific sentiments like positive, neutral, or negative feedback. \

2. For course difficulty analysis, ensure to consider both the course_difficulty and study_hours (per week) columns to provide insight into the relationship between course difficulty and study effort.\

"""