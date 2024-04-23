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

llamaparse_api_key = "llx-15it5aindiMHkKwbasNKUWxMp7y6YsCBZouF6mGz4LeseMAz"
groq_api_key = "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"

client = Groq(
    #api_key=st.secrets["GROQ_API_KEY"],
    api_key="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"
)
def load_or_parse_data(file_path):
    parsing_instruction = """Ce document présente les équipements.
    Quand je te pose une question, renvoie-moi juste la partie du document qui répond à la question, ne cherche pas à reformuler,
    renvoie juste la portion du document qui répond à la question"""
    parser = LlamaParse(
        api_key=llamaparse_api_key,
        result_type="markdown",
        parsing_instruction=parsing_instruction,
        max_timeout=5000,
    )
    llama_parse_documents = parser.load_data(file_path)

    return llama_parse_documents

def create_vector_database(file_path):
    llama_parse_documents = load_or_parse_data(file_path)

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
    st.set_page_config(page_icon="💬", layout="wide", page_title="Mafoya1er Goes Brrrrrrrr...")
    
    st.title("Document QA Chatbot")
    st.sidebar.title("Upload Document")
    
    uploaded_file = st.sidebar.file_uploader("Upload a file", type=["pdf", "md"])

    if uploaded_file is not None:
        file_path = os.path.join("uploads", uploaded_file.name)
        os.makedirs("uploads", exist_ok=True)  # Create the 'uploads' directory if it doesn't exist
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.sidebar.success('File uploaded successfully!')
        vector_store, embed_model = create_vector_database(file_path)

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

        #query = st.text_input("Enter your question:")
        #if st.button("Ask"):
            

        #st.write("Response:", response['result'])
    
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

            # Display user prompt and assistant responses
            for message in st.session_state.messages:
                avatar = "🤖" if message["role"] == "assistant" else "👨‍💻"
                with st.chat_message(message["role"], avatar=avatar):
                    st.markdown(message["content"])

            # Use the generator function with st.write_stream for assistant responses
            with st.chat_message("assistant", avatar="🤖"):
                chat_responses_generator = response["result"]
                full_response = st.write(chat_responses_generator)

            # Append the full response to session_state.messages
            if isinstance(full_response, str):
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )
            else:
                # Handle the case where full_response is not a string
                #combined_response = "\n".join(str(item) for item in full_response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )

        except Exception as e:
            st.error(e, icon="🚨")

if __name__ == "__main__":
    main()
