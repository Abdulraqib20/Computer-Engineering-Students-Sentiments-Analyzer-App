DATA_FILE = "./src/data/survey_data.csv"
# relevant_columns: ['course_code', 'feedback', 'previous_experience', 'gender', 'attendance', 
#                   'course_difficulty', 'study_hours (per week)', 'overall_satisfaction', 'department']

RELEVANT_COLUMNS = ['course_code', 'previous_experience', 'gender', 'attendance', 'course_difficulty', 
                  'study_hours (per week)', 'overall_satisfaction', 'department']
# model name
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
# db_uri: 'postgresql://postgres:admin@localhost:5432/cpe'