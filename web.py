import os
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import WebBaseLoader
from langchain.llms import OpenAI

loader = WebBaseLoader("https://www.duncangough.com/portfolio.html")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(texts, embeddings)

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":1})

chain = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0.9), chain_type="stuff", retriever=retriever, input_key="question")

st.title("LLMs & Web data")
st.info('Website: https://duncangough.com/portfolio.html')

url = st.text_input("### Ask a question 🔍", '')
print(url)

if(st.button("Submit")):
    with st.spinner("Processing..."):
        response = chain({"question": url})
        print(response)
    
        if response:
            st.success(response['result'])