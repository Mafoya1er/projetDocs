import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
import sys

# Function to load PDF, split into chunks, and create the retrieval chain
import os
import tempfile

# Function to load PDF, split into chunks, and create the retrieval chain
def load_pdf_and_create_chain(pdf_file, api_token):
    # Save the uploaded PDF file locally
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(pdf_file.read())
        pdf_path = tmp_file.name

    # Load the pdf file and split it into smaller chunks
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split the documents into smaller chunks 
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # We will use HuggingFace embeddings 
    embeddings = HuggingFaceEmbeddings()

    # Using Chroma vector database to store and retrieve embeddings of our text
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever(search_kwargs={'k': 2})

    # We are using Mistral-7B for this question answering 
    repo_id = "mistralai/Mistral-7B-v0.1"
    llm = HuggingFaceHub(huggingfacehub_api_token=api_token, 
                         repo_id=repo_id, model_kwargs={"temperature":0.1, "max_new_tokens":50})

    # Create the Conversational Retrieval Chain
    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever,return_source_documents=True)

    # Delete the temporary file
    os.unlink(pdf_path)

    return qa_chain


# Main function
def main():
    st.title("Question Answering System")
    
    # File uploader for PDF
    st.sidebar.title("Upload PDF")
    pdf_file = st.sidebar.file_uploader("Upload PDF", type=['pdf'])

    # API token input
    api_token = st.sidebar.text_input("Enter Hugging Face API Token")

    # If PDF is uploaded and API token is provided, load the PDF and create the chain
    if pdf_file is not None and api_token:
        qa_chain = load_pdf_and_create_chain(pdf_file, api_token)
        chat_history = []
        # Input prompt for questions
        question = st.text_input("Enter your question here:")
        
        # Button to ask the question
        if st.button("Ask"):
            if question:
                result = qa_chain({'question': question, 'chat_history': chat_history})
                st.write(f"Answer: {result['answer']}")
                chat_history.append((question, result['answer']))
            else:
                st.warning("Please enter a question.")

# Entry point of the application
if __name__ == "__main__":
    main()
