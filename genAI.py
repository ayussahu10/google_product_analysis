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


# def generate_word_cloud(word_list):
#     text = " ".join(word_list)
#     wordcloud = WordCloud(width=500, height=300, background_color='white').generate(text)
#     fig=plt.figure(figsize=(10, 5))
#     fig=plt.imshow(wordcloud, interpolation='bilinear')
#     fig=plt.axis('off')
#     st.header("Trending hashtags")
#     st.pyplot()


def major_topic_discussion(docs,llm):

    map_prompt = """What is the major topic of discussion: 
    "{text}"
    Provide full summary:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

    summary_chain1 = load_summarize_chain(llm=llm,
                                     chain_type='map_reduce',
                                     map_prompt=map_prompt_template,
                                     return_intermediate_steps=True
                                    )
    
    output = summary_chain1(docs)
    content5=output["intermediate_steps"]

    def major(content):
        model=TextGenerationModel.from_pretrained(model_name='text-bison@001')
        instruction = """ summarize the major topic of discussion based on the content.
        """
        result=model.predict(f'''{instruction},
                        content:{content} 
                        ''',**parameters)
        data=result.text
        return data
    
    res5=major(content5)
    st.header("Major Topic of Discussion")
    return res5

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
    map_prompt = """ Top 5 positive reviews based on the customer reviews : 
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
    content2=output["intermediate_steps"]
    def positive(content):
        model=TextGenerationModel.from_pretrained(model_name='text-bison@001')
        instruction = """ summarize and list only the top 5 positive points based on the content.
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
    map_prompt = """ top 5 negative reviews based on the customer tweets : 
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
        instruction = """ List only the top 5 negative points based on the content.
        """
        result=model.predict(f'''{instruction},
                        content:{content} 
                        ''',**parameters)
        data=result.text
        return data
    
    res3=negative(content3)
    
    st.markdown("<center><h1 style='font-size: 20px; text-decoration: underline;'> Company's focus areas</h1></center>", unsafe_allow_html=True)
    return res3


def pain_points(docs,llm):
    map_prompt = """ top 5 complaints or painpoints of the customer and rank by priority based on the customer reviews : 
    "{text}"
    Provide full summary:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

    summary_chain1 = load_summarize_chain(llm=llm,
                                     chain_type='map_reduce',
                                     map_prompt=map_prompt_template,
                                     return_intermediate_steps=True
                                    )
    
    output = summary_chain1(docs)

    content4=output["intermediate_steps"]

    def painpoints(content):
        model=TextGenerationModel.from_pretrained(model_name='text-bison@001')
        instruction = """ List only the top 5 pain points based on the content. Display it in a ordered list.
        """
        result=model.predict(f'''{instruction},
                        content:{content} 
                        ''',**parameters)
        data=result.text
        return data

    res4=painpoints(content4)
    st.header("Pain Points")
    return res4


def plot_pie_chart(data_dict):
    labels = list(data_dict.keys())
    values = list(data_dict.values())
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    fig.update_layout(
    margin=dict(t=0, b=0),xaxis_title='Hastags', yaxis_title='Count',height=300  # Set top and bottom margin to 0
    )
    st.markdown('<center><h1 style="font-size: 20px; text-decoration: underline;">Trending hashtags</h1></center>', unsafe_allow_html=True)

    st.plotly_chart(fig, use_container_width=True)

def trending_hashtags(docs,llm):   
    map_prompt = """ Extract all the hashtags: 
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
    content_hash=output["intermediate_steps"]
    hash1=[]
    
    def extract_hashtags(input_str):
        # Define the regex pattern to match words starting with '#'
        pattern = r'\#\w+'
        
        # Use findall() to extract all occurrences of the pattern in the input string
        hashtags_list = re.findall(pattern, input_str)
        
        return hashtags_list
    
    for i in range(len(content_hash)): 
        hashtags = extract_hashtags(content_hash[i])
        hash1.append(hashtags)
    flat_list = list(chain(*hash1))

    def count_words(word_list):
        word_count = {}
        for word in word_list:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1
        return word_count

    words = flat_list
    word_count = count_words(words)
    sorted_dict = dict(sorted(word_count.items(), key=lambda item: item[1], reverse=True))
    if len(sorted_dict) > 10:
        top_10_items = sorted(sorted_dict.items(), key=lambda item: item[1], reverse=True)[:10]
        sorted_dict = dict(top_10_items)
    plot_pie_chart(sorted_dict)
        
parameters = {

    "temperature": 0,

    "max_output_tokens": 1024,
}
