from langchain.embeddings.vertexai import VertexAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import VertexAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.summarize import load_summarize_chain 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import PromptTemplate
from vertexai.preview.language_models import TextGenerationModel, ChatModel, CodeGenerationModel
import re
import os
import streamlit as st
import time
import plotly.graph_objs as go
from itertools import chain
from langchain import PromptTemplate
from itertools import chain
from wordcloud import WordCloud



timestamp = time.strftime("%Y%m%d%H%M%S")
filename = f"image_{timestamp}.png"

st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache_data

def data(product_tweets):
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([product_tweets])
    return docs

def summarization(docs,llm):

    map_prompt = """ Write a summary of about 200 words based on the reviews: 
    "{text}"
    Provide full summary instead of concise summary:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

    summary_chain1 = load_summarize_chain(llm=llm,
                                     chain_type='map_reduce',
                                     map_prompt=map_prompt_template,
                                     return_intermediate_steps=True
                                    )
    
    output = summary_chain1(docs)
    content3=output["intermediate_steps"]

    def summary1(content):
        model=TextGenerationModel.from_pretrained(model_name='text-bison@001')
        instruction = """ Summarize the content of about 200 words.
        """
        result=model.predict(f'''{instruction},
                        content:{content} 
                        ''',**parameters)
        data=result.text
        return data
    
    res3=summary1(content3)
    st.markdown('<h1 style="font-size: 25px; text-decoration: underline;">Summary</h1>', unsafe_allow_html=True)
    return res3

def positive_reviews(docs,llm):
    map_prompt = """ Top 5 positive reviews for the product based on the customer review.
                    "{text}"
                    Provide full summary instead of concise summary.Make sure the lines are repeated
                 """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
    summary_chain1 = load_summarize_chain(llm=llm,
                                     chain_type='map_reduce',
                                     map_prompt=map_prompt_template,
                                     return_intermediate_steps=True
                                    )
    
    output = summary_chain1(docs)
    content2=output["intermediate_steps"]
    def positive(content):
        model=TextGenerationModel.from_pretrained(model_name='text-bison@001')
        instruction = """summarize and list the top 5 positive reviews on security areas, performance, user interface based on the content.
                     seggragate the reviews under these categories Security, Performance, User Interface.
                     Do not repeat. Display 5 points under each category in an ordered list.
                     Follow this format.
Security
*
*
*
*
*
Performance
*
*
*
*
*
User Interface
*
*
*
*
*
                      """
        result=model.predict(f'''{instruction},
                        content:{content} 
                        ''',**parameters)
        data=result.text
        return data
    
    res2=positive(content2)
    st.markdown('<center><h1 style="font-size: 20px; text-decoration: underline;">Positive reviews about product/services</h1></center>', unsafe_allow_html=True)
    return res2

def negative_reviews(docs,llm):
    map_prompt = """top 5 focus areas for the product based on the customer review: 
                    "{text}"
                    Provide full summary instead of concise summary:
                """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

    summary_chain1 = load_summarize_chain(llm=llm,
                                     chain_type='map_reduce',
                                     map_prompt=map_prompt_template,
                                     return_intermediate_steps=True
                                    )
    
    output = summary_chain1(docs)
    content3=output["intermediate_steps"]

    def negative(content):
        model=TextGenerationModel.from_pretrained(model_name='text-bison@001')
        instruction = """ top 5 focus areas for the product based on the customer reviews, 
        seggragate the reviews under these categories Security, Performance, User Interface.
        Do not repeat. Display it in a ordered list.
        Follow this format.
Security
*
*
*
*
*
Performance
*
*
*
*
*
User Interface
*
*
*
*
*
        """
        result=model.predict(f'''{instruction},
                        content:{content} 
                        ''',**parameters)
        data=result.text
        return data
    
    res3=negative(content3)
    
    st.markdown("<center><h1 style='font-size: 20px; text-decoration: underline;'> Company's focus areas</h1></center>", unsafe_allow_html=True)
    return res3


parameters = {

    "temperature": 0,

    "max_output_tokens": 1024,
}
