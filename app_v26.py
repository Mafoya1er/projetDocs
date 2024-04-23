import streamlit as st
from typing import Generator
from groq import Groq
from llama_parse import LlamaParse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import os
#
import joblib
import shutil

llamaparse_api_key = "llx-15it5aindiMHkKwbasNKUWxMp7y6YsCBZouF6mGz4LeseMAz"
groq_api_key = "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"

client = Groq(
    #api_key=st.secrets["GROQ_API_KEY"],
    api_key="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"
)

st.set_page_config(page_icon="💬", layout="wide", page_title="Mafoya1er Goes Brrrrrrrr...")

def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )

icon("🏎️")

st.subheader("Mafoya1er Chat App", divider="rainbow", anchor=False)

def load_or_parse_data(file_path):
 
   
    parsing_instruction = """Ce document est un CV , il présente le parcours académique d'un candidat, ses compétences, ses expériences, ses diplômes.
    Il présente aussi ces loisirs , ses centres intérêts et toutes informations personnelles du candidat. Tu vas utiliser ce document pour repondre à toutes les questions qui te seront posées """
    parser = LlamaParse(
        api_key=llamaparse_api_key,
        result_type="markdown",
        parsing_instruction=parsing_instruction,
        max_timeout=5000,
    )
    llama_parse_documents = parser.load_data(file_path)
    # Save the parsed data to a file
    print("Saving the parse results in .pkl format ..........")
    #joblib.dump(llama_parse_documents, data_file)

        # Set the parsed data to the variable
       
    return  llama_parse_documents

def create_vector_database(file_path):
    llama_parse_documents = load_or_parse_data(file_path)

    # Check if there are documents to process
    if not llama_parse_documents:
        return None, None  # Return None if no documents are found

    with open('data/output.md', 'a') as f:
        for doc in llama_parse_documents:
            f.write(doc.text + '\n')

    markdown_path = "data/output.md"
    loader = UnstructuredMarkdownLoader(markdown_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    vs = Chroma.from_documents(
        documents=docs,
        embedding=embed_model,
        persist_directory="chroma_db_llamaparse1",
        collection_name="rag"
    )

    return vs, embed_model

def main():
    
    st.sidebar.title("Upload Document")
    
    uploaded_file = st.sidebar.file_uploader("Upload a file", type=["pdf", "md"])
    #shutil.rmtree("uploads")

    if uploaded_file is not None:
        # Supprimer les fichiers du dossier data
        data_folder = 'data'
        if os.path.exists(data_folder):
            shutil.rmtree(data_folder)
        os.makedirs(data_folder, exist_ok=True)  # Recréer le dossier data

        # Supprimer les fichiers du dossier chroma_db_llamaparse1
        chroma_folder = 'chroma_db_llamaparse1'
        if os.path.exists(chroma_folder):
            shutil.rmtree(chroma_folder)
        shutil.rmtree("uploads")

        # Créer un nouveau dossier chroma_db_llamaparse1
        os.makedirs(chroma_folder, exist_ok=True)    

    if uploaded_file is not None:
        file_path = os.path.join("uploads", uploaded_file.name)
        os.makedirs("uploads", exist_ok=True)  # Create the 'uploads' directory if it doesn't exist
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.sidebar.success('File uploaded successfully!')
        vector_store, embed_model = create_vector_database(file_path)

        if vector_store is None:
            st.error("No documents found in the uploaded file. Please upload a valid document.")
            return

        chat_model = ChatGroq(
            temperature=0,
            model_name="mixtral-8x7b-32768",
            api_key=groq_api_key,
        )

        retriever = vector_store.as_retriever(search_kwargs={'k': 3})

        custom_prompt_template = """Use the following pieces of information to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}
        Question: {question}

        Only return the helpful answer below and nothing else.
        Helpful answer:
        """

        prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

        qa = RetrievalQA.from_chain_type(
            llm=chat_model,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Define model details
    model_option = "mixtral-8x7b-32768"
    max_tokens_range = 32768  # Fixed token size for the Mistral model

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        avatar = "🤖" if message["role"] == "assistant" else "👨‍💻"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # Chat input and conversation
    if prompt := st.chat_input("Enter your prompt here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar="👨‍💻"):
            st.markdown(prompt)

        try:
            # Fetch response from Groq API
            response = qa.invoke({"query": prompt})

            # Display user prompt
            with st.chat_message("user", avatar="👨‍💻"):
                st.markdown(prompt)

            # Use the generator function with st.write_stream for assistant responses
            with st.chat_message("assistant", avatar="🤖"):
                chat_responses_generator = response["result"]
                st.write(chat_responses_generator)

            # Append the full response to session_state.messages
            st.session_state.messages.append(
                {"role": "assistant", "content": chat_responses_generator}
            )

        except Exception as e:
            st.error(e, icon="🚨")

if __name__ == "__main__":
    main()
