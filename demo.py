# conda create -p venv python==3.10

# conda activate venv/

# Get Gemini api key from https://makersuite.google.com/app/apikey and put it in the .env file

# pip install requirements.txt

# Run the command: streamlit run app.py

 

import os

import streamlit as st

import graphviz as graphviz

 

# Document Loading

#from langchain.document_loaders import PyPDFLoader , DirectoryLoader #Depreciated

from langchain_community.document_loaders import PyPDFLoader , DirectoryLoader

from PyPDF2 import PdfReader

from langchain_community.document_loaders import UnstructuredPDFLoader

from pdfminer import psparser

 

# Text Chunking

from langchain.text_splitter import RecursiveCharacterTextSplitter

 

# Embeddings

from langchain_huggingface import HuggingFaceEmbeddings

#from langchain.embeddings import HuggingFaceEmbeddings # Depreciated

#from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_google_genai import GoogleGenerativeAIEmbeddings

#from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

 

# Vector DB

#from langchain.vectorstores import FAISS # Depreciated

from langchain_community.vectorstores import FAISS

 

# Retrivers

from langchain.chains import RetrievalQA

from langchain.chains import conversational_retrieval

#from langchain.retrievers import BM25Retriever # Depreciated

from langchain.retrievers import EnsembleRetriever

# from langchain_community.retrievers import BM25Retriever

# from langchain.retrievers import BM25Retriever

from langchain_core.documents import Document

from langchain.retrievers import ContextualCompressionRetriever

 

# Reranker

from langchain.retrievers.document_compressors import CohereRerank

from langchain.retrievers.document_compressors import FlashrankRerank

from flashrank import Ranker,RerankRequest

 

# Promts

#from langchain import PromptTemplate

from langchain.prompts import PromptTemplate

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnablePassthrough

 

# Buffer Memory

from langchain.memory import ConversationBufferWindowMemory

from langchain.memory import ConversationBufferMemory

 

# LLM

from langchain_google_genai import ChatGoogleGenerativeAI

import google.generativeai as genai

#from langchain.llms import Transformers

# from langchain_huggingface.llms import HuggingFacePipeline

# from sentence_transformers import SentenceTransformer

# from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

 

# Chains

from langchain.chains import LLMChain

 

 

import joblib

 

#Environment Loading

from dotenv import load_dotenv

import google.generativeai as genai

#from pdf_project.helper import *

#print("Libraries Loaded")

 

# -----------------------------------------------------------------------------------------------

 

# -----------------------------------------------------------------------------------------------

 

 

 

# import asyncio

 

# # Ensure event loop is setup

# try:

#     loop=asyncio.get_event_loop()

# except RuntimeError:

#     loop=asyncio.new_event_loop()

#     asyncio.set_event_loop(loop)

 

 

 

load_dotenv()

os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

 

os.getenv("HUGGING_FACE_KEY")

genai.configure(api_key=os.getenv("HUGGING_FACE_KEY"))

 

 

# def load_pdf(pdf_docs):

#     loader=DirectoryLoader(pdf_docs,glob="*.pdf",loader_cls=PyPDFLoader)

#     text=loader.load()

#     return text

 

 

def get_pdf_text(pdf_docs):

    text=""

    for pdf in pdf_docs:

        pdf_reader= PdfReader(pdf)

        for page in pdf_reader.pages:

            text+= page.extract_text()

    return  text

 

 

 

def get_text_chunks(text):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50,separators=["\\n\\n", "\\n", " ", ""])

    chunks = text_splitter.split_text(text)

    return chunks

 

 

# def get_vector_store(text_chunks):

#     embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

#     vector_store.save_local("faiss_index")

 

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",model_kwargs={"device":"cpu"})

    #embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    vector_store.save_local("faiss_index")

 

# def Key_word_retriver(text_chunks):

#     keyword_retriever = BM25Retriever.from_documents(text_chunks)

#     keyword_retriever.k =  5

#     joblib.dump(keyword_retriever,"bm25.sav")

 

 

def get_conversational_chain():

 

    prompt_template = """

    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in

    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n.Page number and source can be provided

    Context:\n {context}?\n


    Question: \n{question}\n

 

    Answer:

    """

 

    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro",

                             temperature=0.3,google_api_key=os.getenv("GOOGLE_API_KEY"))

 

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context","question"])

    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

 

    vector_store_f=FAISS.load_local('faiss_index',embeddings,allow_dangerous_deserialization=True)

 

    retriever_vectordb = vector_store_f.as_retriever(search_kwargs={"k": 5})

 

    # keyword_retriever=joblib.load("bm25.sav")

 

    # ensemble_retriever = EnsembleRetriever(retrievers=[retriever_vectordb,keyword_retriever],weights=[0.5, 0.5])

 

    model_name = "ms-marco-MultiBERT-L-12"  # Example model, adjust as needed

    flashrank_client = Ranker(model_name=model_name,cache_dir="/opt")

 

    compressor = FlashrankRerank(client=flashrank_client, top_n=3, model=model_name)

    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever_vectordb)

 

    # mem = ConversationBufferMemory(

    #                                 memory_key="chat_history",

    #                                 max_len=50,

    #                                 return_messages=True,

    #                                 output_key='answer'

    #                             )

   

    # memo = ConversationBufferMemory(

    # memory_key="chat_history",

    # chat_memory=chat_history,  # this is your persistence strategy subclass of `BaseChatMessageHistory`

    # output_key="answer",

    # return_messages=True

    # )

    # Define memory to track conversation history
    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

 

    conversation = RetrievalQA.from_chain_type(

        llm = model,

        chain_type="stuff",

        retriever=compression_retriever,

        return_source_documents = True,

        verbose=True,

        chain_type_kwargs={"prompt":prompt}
        # ConversationBufferWindowMemory(k=4)

    )

 

    # pp.pprint(chain({'question': q1, 'chat_history': memory.chat_memory.messages}))

    # memory.chat_memory.messages # The chat history/ stored memory can be viewed with

 

    chain= ({'context':compression_retriever, 'question':RunnablePassthrough()}

            |prompt

            |model

            |StrOutputParser())

 

    #chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

 

    return conversation

 

 

 

def user_input(user_question):

    #embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

    # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

   

    # new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

 

    # docs = new_db.similarity_search(user_question)

 

    chain = get_conversational_chain()

 

   

    response = chain.invoke(user_question)

    # result=response['result']

    # source_data=response['source_documents']

 

    st.write("Reply: ", response["result"])

    #st.write("raw",response)

    #st.write("Source",response['source_documents'])

 

    # for d in range(0,len(response['source_documents'])):

    #     meta_data_pdf=response['source_documents'][d].metadata['source']

    #     meta_data_page=response['source_documents'][d].metadata['page']

    #     meta_data_page_content=response['source_documents'][d].page_content

 

    #     st.write("Source_PDF\n",meta_data_pdf)

    #     st.write("Page_No\n",meta_data_page)

    #     st.write("Page_Content",meta_data_page_content)

 

from pathlib import Path

# define a folder to store the uploaded PDF's

 

UPLOAD_FOLDER="data_storage"

 

# Define a folder if it doesn't exist

Path(UPLOAD_FOLDER).mkdir(parents=True,exist_ok=True)

 

def save_uploaded_file(uploaded_file):

    with open(os.path.join(UPLOAD_FOLDER,uploaded_file.name),"wb") as f:

        f.write(uploaded_file.getbuffer())

    return uploaded_file.name

 

def clear_uploaded_files():

    for file in os.listdir(UPLOAD_FOLDER):

        os.remove(os.path.join(UPLOAD_FOLDER,file))

    st.success("All files have been deleted.")

 

def show_uploaded_files():

    '''Display a list of all files currently in the upload folder'''

    files=os.listdir(UPLOAD_FOLDER)

    if files:

        st.subheader("Available Files")

        for file in files:

            st.write(file)

    else:

        st.write("No files available")

 

def main():

    st.set_page_config("Chat PDF")

    st.header("Chat and Keyword Search with Multiple PDF using Gemini")

    st.subheader("Progress Bar")

    progress_text="Operation in Progress. Please Wait"

    my_bar=st.progress(30,progress_text)

    st.balloons()

    st.graphviz_chart('''

                      digraph{

                      Upload_PDF_using_Browse_files->Submit

                      Submit->Inferences

                      }

                      ''')

 

    user_question = st.text_input("Ask a Question from the PDF Files")

 

    if user_question:

        user_input(user_question)

        my_bar.progress(100,"Work Done")

 

    with st.sidebar:

        st.title("Menu:")

        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)

 

        if st.button("Submit & Process"):

           

            st.subheader("uploaded files")

            for uploaded_file in pdf_docs:

                file_name=save_uploaded_file(uploaded_file)

                st.write(f"Saved file: {file_name}")

 

 

 

            with st.spinner("Processing..."):

                raw_text = get_pdf_text(pdf_docs)

                text_chunks = get_text_chunks(raw_text)

                get_vector_store(text_chunks)

                #Key_word_retriver(text_chunks)

                my_bar.progress(70,"Document Processing Done")

 

                st.success("Done")

       

        st.button("Show Available Files",on_click=show_uploaded_files)

 

        if st.button("Clear Files"):

            clear_uploaded_files()

            # st.experimental_rerun()
            # Use this to "simulate" a rerun by setting query parameters
            st.experimental_set_query_params(clear=True)
            # st.set_query_params(clear=True)

 

 

 

if __name__ == "__main__":
    main()


