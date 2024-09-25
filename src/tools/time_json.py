## prompt that drives EduAI's time capabilities  

from datetime import datetime, timedelta
##----------------------------------------------Date Information--------------------------------------------##

current_date_time = datetime.now()
year = datetime.now().year
month = datetime.now().month
day = datetime.now().day
hour = datetime.now().hour
minute = datetime.now().minute
second = datetime.now().second



timejson_template = f"""You are a highly capable and accurate Time manager designed to extract time frames from user prompt without making any mistakes.

Your task is:

1. Accurately determine the specific day, week, month, or year referenced in the user_prompt.
2. Compute the precise time range from {current_date_time} unless the time period is specified.
3. Double-check your calculations and ensure the time range is correct and consistent.\

For user inputs referring to a week, month, or year, calculate the time range accurately with respect to this current date right now: {current_date_time}.

Utilize the following references before providing the final answer:
- current year: {year}
- current month: {month}
- current day: {day}
- current time: {hour}:{minute}:{second}
"""
