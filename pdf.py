import os
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI

def load_file_and_setup_chain(filename):
    loader = PyPDFLoader(file_path='./data/pdf/' + filename)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(texts, embeddings)

    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":1})

    return RetrievalQA.from_chain_type(llm=OpenAI(temperature=0.9), chain_type="stuff", retriever=retriever, input_key="question")

st.title("LLMs & PDF data")
uploaded_file = st.file_uploader("Choose .pdf", type=['pdf'])

if uploaded_file is not None:
    with st.spinner("Processing..."):
        chain = load_file_and_setup_chain(uploaded_file.name)

    question = st.text_input("### Search the data 🔍", '')

    if(st.button("Submit")):
        with st.spinner("Processing..."):
            response = chain({"question": question})
        
        if response:
            st.success(response['result'])