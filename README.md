# The Sentiment Analysis App for Computer Engineering Students 

## Description
This project focuses on sentiment analysis of student feedback in engineering education. It is particularly directed at my engineering class in the university. It aims to gain insights from feedback data, improve education quality, and enhance the student experience. The project employs natural language processing techniques, topic modeling, and machine learning algorithms to analyze sentiments expressed by the students. It provides actionable recommendations based on sentiment analysis results.

## Objectives (Aims & Goals of this Project)
 - To uncover sentiments expressed in the feedback and gain a comprehensive understanding of student perceptions, satisfaction and identfying areas of improvement.
 - To ensure real-time analysis to provide immediate insights into prevailing student sentiments.
 - Creating interactive visualizations for dynamic displays of sentiment trends over time.
 - Extracting insights into teaching methodologies, lecturers and departmental courses.
 - Identifying and highlighting specific challenges faced by students for targeted improvements.
 - Facilitating interactive exploration of sentiment analysis results for deeper understanding.
 - Establishing a continuous feedback loop for ongoing improvement in educational practices.
 - Enabling lecturers to download sentiment analysis data for in-depth analysis.
 - Ensuring privacy and ethical handling of student feedback data in compliance with regulations.
- Aiding the lecturers in interpreting and utilizing sentiment analysis results.

## Key Features
1. **Sentiment Analysis Functionality**
       - Utilizes a sentiment analysis model to score feedback text.
       - Analyzes sentiments as positive, neutral, or negative.

    2. **User Input Collection**
       - Gathers user's feedback and related information based on the following:
         - Course code ( The code for the Course)
         - Previous experience (Whether the user has previous experience with the course, lecturer, etc)
         - Gender (The gender of the user)
         - Attendance (The attendance rate of the user)
         - Course difficulty (Perceived difficulty of the course)
         - Study hours per week (Number of hours devoted to studying per week for the course)
         - Overall satisfaction (Metric used to evaluate user's satisfaction with the course, lecturer, teaching, etc)
         - Department (Whether the user belongs to the department of Computer Engineering)
         - Date and time of feedback submission (Date and time of feedback submission)

    3. **Text Preprocessing**
       - Preprocesses the user feedback text using Natural Language Processing techniques.

    4. **Percentage Confidence**
       - Percentage level of how confident the model is making the prediction.

    5. **Interactive Visualization**
       - Provides various interactive plots for visualizing sentiment analysis results and other key metrics.
       - Displays sentiment distribution in various charts (bar chart, pie chart, word cloud).
       - Presents feedback counts based on course difficulty, course code, and gender.
       - Provides insights into word frequency and usage in feedback.
       - Explores the distribution of study hours, word count, and overall satisfaction.

    6. **Summary Statistics**
       - Offers a sentiment summary with counts of positive, neutral, and negative feedback including the percentage confidence results.

    8. **Interactive Exploration**
       - Allows users to trigger the exploration of visualizations by clicking a button.

    9. **Real-Time Feedback Data Access**
        - The app allows users access to real-time feedback data after getting their prediction results.
        - Users can download and view the data directly within the app.

    10.  **Automatic Real-Time Saving**
        - The app works in real-time, automatically saving prediction results and other insights generated.

## Installation
Install the required dependencies: `pip install -r requirements.txt`

## Usage
1. Collect and clean the feedback data.
2. Preprocess the data using text preprocessing techniques.
3. Perform aspect-based sentiment analysis on specific topics or keywords.
4. Apply topic modeling to identify key themes and topics in the feedback.
5. Detect and analyze emotions expressed in the feedback.
6. Generate visualizations to understand sentiment distribution and feedback trends.
7. Build a web-app for the project

## Limitations and Future Work
One of the limitations of this project is the relatively small size of the dataset. The data collected for sentiment analysis on student feedback in engineering education may not represent the entire student population or provide a comprehensive view of sentiments. This limitation could affect the generalizability of the findings and the accuracy of the sentiment analysis results.

To address this limitation, future work could involve collecting a larger and more diverse dataset to improve the robustness and reliability of the sentiment analysis. Additionally, exploring external data sources or incorporating data from multiple educational institutions could provide a broader perspective on student sentiments and enhance the overall analysis.

## Contributing
Contributions are welcome! If you would like to contribute to the project, feel free to reach out to me. Together, we can enhance the analysis and make a positive impact on engineering education.

## Contact
For any questions or inquiries, please contact abdulraqibshakir03@gmail.com.
