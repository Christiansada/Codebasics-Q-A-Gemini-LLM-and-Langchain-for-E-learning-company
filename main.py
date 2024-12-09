import streamlit as st
from langchain_helper import get_qa_chain

st.title("Codebasics Q&A 🌱")


question = st.text_input("Question: ")

if question:
    chain = get_qa_chain()
    response = chain.invoke(question)

    st.header("Answer")
    st.write(response["result"])






