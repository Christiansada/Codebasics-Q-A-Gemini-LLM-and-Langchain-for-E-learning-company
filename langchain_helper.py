from langchain_google_genai import GoogleGenerativeAI;
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


def get_qa_chain():
    
    llm = GoogleGenerativeAI(model='gemini-pro', google_api_key= 'AIzaSyCc8-ADN15Yb4CMLT_VRezvmIUu2SOxYxc') 


    loader = CSVLoader(file_path='codebasics_faqs.csv', source_column="prompt")

    # Store the loaded data in the 'data' variable
    data = loader.load()

    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    mpnet_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )



    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=data,
                                    embedding=mpnet_embeddings)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold = 0.7)



    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""


    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}




    chain = RetrievalQA.from_chain_type(llm=llm,
                                chain_type="stuff",
                                retriever=retriever,
                                input_key="query",
                                return_source_documents=True,
                                chain_type_kwargs=chain_type_kwargs)
    return chain


if __name__ == "__main__":
    chain = get_qa_chain()
    print(chain.invoke("Do you have javascript course?"))
