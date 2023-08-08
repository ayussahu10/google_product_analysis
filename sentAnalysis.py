import pandas as pd
import textblob
import re
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st

def sentiment_analysis(df):

    def data_processing(text):
        text = text.lower()
        text = re.sub(r"https\S+|www\S+https\S+", '',text, flags=re.MULTILINE)
        text = re.sub(r'\@w+|\#','',text)
        text = re.sub(r'[^\w\s]','',text)
        text_tokens = word_tokenize(text)
        filtered_text = [w for w in text_tokens if not w in stop_words]
        return " ".join(filtered_text)

    text_df=df
    text_df.text = text_df['Title'].apply(data_processing)
    
    stemmer = PorterStemmer()
    def stemming(data):
        text = [stemmer.stem(word) for word in data]
        return data
    text_df['text'] = text_df['text'].apply(lambda x: stemming(x))

    def polarity(text):
        return TextBlob(text).sentiment.polarity
    text_df['polarity'] = text_df['text'].apply(polarity)
 
    def sentiment(label):
        if label <0:
            return "Negative"
        elif label ==0:
            return "Neutral"
        elif label>0:
            return "Positive"
    
    text_df['sentiment'] = text_df['polarity'].apply(sentiment)

    # Get the value counts of the 'Category' column
    value_counts = df['sentiment'].value_counts()

    # Create a pie chart using Plotly
    fig = go.Figure(data=[go.Pie(labels=value_counts.index, values=value_counts.values)])
    fig.update_layout(
    margin=dict(t=0, b=0),xaxis_title='Sentiments', yaxis_title='Distribution count',height=300  # Set top and bottom margin to 0
    )
    st.markdown('<center><h1 style="font-size: 20px; text-decoration: underline;">Overall sentiment distribution</h1></center>', unsafe_allow_html=True)
    st.markdown(
    """
    <style>
    .plotly .js-plotly-plot {{
        border: 1px solid #d3d3d3;
        border-radius: 5px;
        padding: 10px;
    }}
    </style>
    """,
    unsafe_allow_html=True
    
    )

    # Set the chart title
    st.plotly_chart(fig, use_container_width=True)

# def tweetsperday(df):
#     df['count'] = 1
#     data_filtered = df[['dates', 'count']]
#     df_tweets_daily = data_filtered.groupby(["dates"]).sum().reset_index()
    
#     categories = df_tweets_daily['dates']
#     values = df_tweets_daily['count']
#     st.markdown('<center><h1 style="font-size: 20px; text-decoration: underline;">Tweets per day</h1></center>', unsafe_allow_html=True)

#     # Create a bar chart using plotly.express.bar
#     fig = px.bar(x=categories, y=values, labels={'x': 'Dates', 'y': 'Tweet Counts'})
#     fig.update_layout(
#     margin=dict(t=0, b=0),height=300  # Set top and bottom margin to 0
#     )
#     # Show the plot
#     st.plotly_chart(fig, use_container_width=True)

def load_data(df):
    data = df
    data['created_at'] = pd.to_datetime(data['created_at'])
    return data  