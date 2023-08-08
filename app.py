import nltkmodules
import streamlit as st
import folium
import math
from streamlit_folium import folium_static
from streamlit_option_menu import option_menu
from pathlib import Path
import base64
import vertexai
import pandas as pd
from langchain.llms import VertexAI
from langchain.chat_models import ChatVertexAI
import re
import time
from vertexai.preview.language_models import TextGenerationModel, ChatModel, CodeGenerationModel
import db_dtypes
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import random
#from retry import retry
import plotly.express as px
import plotly.graph_objs as go
from langchain.embeddings.vertexai import VertexAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.summarize import load_summarize_chain 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import PromptTemplate
from itertools import chain
from wordcloud import WordCloud
import os
import numpy as np
import seaborn as sns
from matplotlib import style
style.use('ggplot')
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import os
import genAI, sentAnalysis

#genAI.py
#SentAnalysis

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

def img_to_html(img_path):
    img_html = "<img src='data:image/png;base64,{}' height='35' width='140'>".format(
        img_to_bytes(img_path)
    )
    return img_html
        
def main():
    st.set_page_config(page_title = "Social media sentiment analysis",layout="wide")
   
    st.markdown(
        f"""
    <style>
        .appview-container .main .block-container {{
            padding-top: 0;
            margin: 0;
            height: 98%;
        }}
        
        .stButton {{
            padding-bottom: 30px;
            
            }}
    </style>
    """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <style>
         div.css-1b2d4l5.esravye1{{
             position: relative;
             bottom: 6px;
             }}
        </style>
        """,
            unsafe_allow_html=True,
        )


    st.markdown(
        f"""
        <style>
         div.css-1b2d4l5.e1f1d6gn1{{
             position: relative;
             bottom: -28px;
             }}
        </style>
        """,
            unsafe_allow_html=True,
        )
    
    st.markdown(
    f"""
    <style>
    .plotly-graph {{
        border: 1px solid #d3d3d3;
        border-radius: 5px;
        padding: 10px;
    }}
    </style>
    """
    , unsafe_allow_html=True
    )

    st.markdown(
    f"""
    <style>
        .div.user-select-none.svg-container{{
            height: 200px;
            
             
    }}
    </style>
    
    """,
    unsafe_allow_html=True
    )
    st.markdown(
    """
    <style>
    .box {
        border: 2px solid #000000; /* Border color (black in this case) */
        padding: 10px; /* Optional padding around the content */
        border-radius: 5px; /* Optional border radius for rounded corners */
    }
    </style>
    """,
    unsafe_allow_html=True
    )

    st.text("")

    st.markdown(
        """
    <style>
        .background {{
            background-color: rgb(241, 237, 238);
            padding: 10px;
            margin-top: 1%;
            border: 1px solid #ccc;
            box-shadow: 4px 4px 5px rgba(0, 0, 0, 0.3);
        }}

        .title_heading {{
            color: #000000;
            font-size: 22px;
            font-weight: bold;
            font-family: "Open Sans", sans-serif;
        }}

        .title {{
            margin-top: 20px;
            display: flex;
        }}

        .button-inline {{
            color: green;
            background-color: rgb(241, 237, 238);
            padding: 10px 20px;
            font-size: 11px;
            font-weight: bold;
            border: 1px solid white;
            margin-left: auto;
            margin-right: 10px;
            height: 20px;
            margin-top: 1px;
            line-height: 0.3;
            border-radius: 5px;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
        }}

        .vertical-bar {{
            display: inline-block;
            height: 1em;
            vertical-align: middle;
        }}
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <style>
        .background_black {{
            background-color: #000000;
            padding-top: 0px;
            border: 1px solid #ccc;
            box-shadow: 4px 4px 5px rgba(0, 0, 0, 0.3);
            margin-top: 2%;
            margin-bottom: -3%;
            position: relative;
        }}

        .paragraph_heading {{
            color: rgb(134, 188, 37);
            font-size: 18px;
            font-weight: bold;
            font-family: "Open Sans", sans-serif;
        }}

        .paragraph_body {{
            color: #ffffff;
            font-size: 14px;
            font-weight: bold;
            font-family: "Open Sans", sans-serif;
        }}

        .paragraph {{
            margin-left: 20px;
            margin-top: 10px;
        }}

        .image {{
            position: absolute;
            top: 8;
            right: 0;
            margin-left: 10px;
        }}
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <style>
        div.css-1vbkxwb.eqr7zpz4 {{
            color: green;
            margin-top: 10%;
            text-align: center;
        }}

        .css-1vbkxwb.eqr7zpz4 p {{
            margin-bottom: 8px;
            font-size: 13px;
            font-weight: bold;
        }}
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <style>
        button.css-1n543e5.e1ewe7hr5 {{
            padding: 2px 2px 2px 2px;
            border: 1px solid #ccc;
            box-shadow: 4px 4px 5px rgba(0, 0, 0, 0.3);
            height: 60%;
            width: 6%;
            text-align: center;
        }}
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <style>
        div.css-12ttj6m.en8akda1 {{
            border: 1px solid #ccc;
            box-shadow: 4px 4px 5px rgba(0, 0, 0, 0.3);
        }}
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <style>
        input.st-be.st-bz.st-c0.st-c1.st-c2.st-c3.st-c4.st-c5.st-c6.st-c7.st-c8.st-b8.st-c9.st-ca.st-cb.st-cc.st-cd.st-ce.st-cf.st-ae.st-af.st-ag.st-ch.st-ai.st-aj.st-by.st-ci.st-cj.st-ck {{
            background-color: rgb(241, 237, 238);
        }}
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <style>
        input.st-bd.st-by.st-bz.st-c0.st-c1.st-c2.st-c3.st-c4.st-c5.st-c6.st-c7.st-b8.st-c8.st-c9.st-ca.st-cb.st-cc.st-cd.st-ce.st-cf.st-ae.st-af.st-ag.st-cg.st-ai.st-aj.st-bx.st-ch.st-ci.st-cj {{
            background-color: rgb(241, 237, 238);
        }}
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <style>
        textarea.st-bd.st-by.st-bz.st-c0.st-c1.st-c2.st-c3.st-c4.st-c5.st-c6.st-c7.st-b8.st-c8.st-c9.st-ca.st-cb.st-cp.st-cq.st-cr.st-cs.st-ae.st-af.st-ag.st-cg.st-ai.st-aj.st-bx.st-ch.st-ci.st-cj.st-ct.st-cu.st-cv {{
            background-color: rgb(241, 237, 238);
        }}
    </style>
    """,
        unsafe_allow_html=True,
    )
    hide_st_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)


    # Define a CSS style for the buttons
    button_style = """
        <style>
            .equal-width-button button {
                width: 200px;
                box-sizing: border-box;
            }
        </style>
    """
    st.markdown(button_style, unsafe_allow_html=True)
    


    st.markdown("""
        <style>
        
            .background {
            background-color: rgb(241, 237, 238);
            padding: 10px;
            margin-top: -120px;
            border: 1px solid #ccc;
            box-shadow: 4px 4px 5px rgba(0, 0, 0, 0.3);
            }
            
            .title_heading {
            color: #000000;
            font-size: 22px;
            font-weight: bold;
            font-family: "Open Sans", sans-serif;
            }
            .title {
            margin-top: 20px;
            display: flex;
            }
            .button-inline {
            color: green;
            background-color: rgb(241, 237, 238);
            padding: 10px 20px;
            font-size: 11px;
            font-weight: bold;
            border: 1px solid white;
            margin-left: auto;
            margin-right: 10px;
            height: 20px;
            margin-top: 1px;
            line-height: 0.3;
            border-radius: 5px;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
            }

            .vertical-bar {
            display: inline-block;
            height: 1em;
            vertical-align: middle;
        }
            
            """
            
        f"""</style>
        <div class="background">
            <p class="title">
            {img_to_html('deloitte_logo.png')}
            <span class ="title_heading"> | Generative AI</span>
            <button class="button-inline" type="button">Logout</button>
        </p>
        </div>
            """,

                unsafe_allow_html=True,

                )
    ##Create a text container with a black background
    st.markdown("""
        <style>
        
            .background_black {
            background-color: #000000;
            padding-top: 0px;
            border: 1px solid #ccc;
            box-shadow: 4px 4px 5px rgba(0, 0, 0, 0.3);
            margin-top: -2%;
            margin-bottom: -3%;
            position: relative;
            }
            
            .paragraph_heading {
            color: rgb(134, 188, 37);
            font-size: 18px;
            font-weight: bold;
            font-family: "Open Sans", sans-serif;
            }
            
            .paragraph_body {
            color: #ffffff;
            font-size: 14px;
            font-weight: bold;
            font-family: "Open Sans", sans-serif;
            }
            .paragraph {
            margin-left: 20px;
            margin-top: 10px;
            }
            .image{
            position: absolute;
            top: 8;
            right: 0;
            margin-left: 10px;
            }

            
        </style>
        <div class="background_black">
        <p class="paragraph">
            <span class ="paragraph_heading">Social media sentiment analysis</span><br>
            <span class ="paragraph_body">A generative AI powered tool that summarizes the reviews of the product based on the customer reviews</span>
            
        </p>
        </div>
            """,

                unsafe_allow_html=True,

                )
    st.markdown("---")

    if "key" not in st.session_state:
        st.session_state.key=False
    
    if "key2" not in st.session_state:
        st.session_state.key2=False
        
        
    c1,c2 = st.columns([6,1])
    with c1:
        revs = ""
        product = st.radio(
        "Select a product to see the analysis",
        ('AndroidOS', 'Chrome', 'Pixel'))

        if product == 'AndroidOS':
            revs='android.txt'
            df = pd.read_csv(r"Android.csv", encoding ="latin-1",engine='python' )
        elif product == 'Chrome':
            revs='chrome.txt'
            df = pd.read_csv(r"Chrome.csv", encoding ="latin-1",engine='python' )
        elif product == 'Pixel':
            revs='googlepixel.txt'
            df = pd.read_csv(r"GooglePixel.csv", encoding ="latin-1",engine='python' )
        else:
            st.write("Please select a product.")
        
        product_revs = ""
        if not os.path.exists(revs):
            st.error(f"File not found: {revs}")
        else:
            with open(revs, 'r') as file:
                product_revs = file.read()
        
        docs=genAI.data(product_revs)
        llm=VertexAI(model='chat-bison@001')  
        
    with c2:
        generate_response = st.button("Submit")

    if generate_response or st.session_state.key:
        st.session_state.key=True
        tab1, tab2 = st.tabs(["Analysis", "Q&A"])
        with tab1:
            summary=genAI.summarization(docs,llm)
            st.markdown(f"""<div style= "text-align: justify;">{summary}</div>""",unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                pr=genAI.positive_reviews(docs,llm)
                st.markdown(f"""<div class="box" >{pr}</div>""", unsafe_allow_html=True)
            with col2:
                nr=genAI.negative_reviews(docs,llm)
                st.markdown(f"""<div class="box">{nr}</div>""", unsafe_allow_html=True)
            with col3:
                sentAnalysis.sentiment_analysis(df)    
            col4 = st.columns(1)  
            with col4:
                genAI.trending_hashtags(docs,llm) 
        
       

if __name__ == "__main__":
    main()  
