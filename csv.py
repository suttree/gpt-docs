import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# Load from CSV
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

def load_file_and_setup_chain(filename):
    print(filename)
    #loader = CSVLoader(file_path='./data/Greater_than___500_Qtr_3_Oct_to_Dec_22_23_Final_for_Publishing.csv')
    #loader = CSVLoader(file_path='./data/Property_assets.csv')
    loader = CSVLoader(file_path='./data/csv/' + filename, encoding='utf-8')

    # Create an index using the loaded documents
    index_creator = VectorstoreIndexCreator()
    #index_creator = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": "db"})
    docsearch = index_creator.from_loaders([loader])

    # Create a question-answering chain using the index
    return RetrievalQA.from_chain_type(llm=OpenAI(temperature=0.9), chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question")


st.title("LLMs & CSV data")
uploaded_file = st.file_uploader("Choose .csv", type=['csv'])

if uploaded_file is not None:
    with st.spinner("Processing..."):
        chain = load_file_and_setup_chain(uploaded_file.name)
        print(uploaded_file)

    question = st.text_input("### Search the data 🔍", '')

    if(st.button("Submit")):
        with st.spinner("Processing..."):
            response = chain({"question": question})
            print(response)
        
            if response:
                st.success(response['result'])