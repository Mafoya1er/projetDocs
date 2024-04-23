import streamlit as st
from llama_parse import LlamaParse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import joblib
import os
import nest_asyncio
from docx import Document
import PyPDF2

# Appliquer nest_asyncio pour gérer les événements asynchrones dans Streamlit
nest_asyncio.apply()

llamaparse_api_key = "llx-15it5aindiMHkKwbasNKUWxMp7y6YsCBZouF6mGz4LeseMAz"
groq_api_key = "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"

@st.cache(allow_output_mutation=True)
def load_or_parse_data(file_path):
    data_file = "./data/parsed_data.pkl"

    if os.path.exists(data_file):
        # Charger les données analysées depuis le fichier
        parsed_data = joblib.load(data_file)
    else:
        # Performer l'étape d'analyse et stocker le résultat dans llama_parse_documents
        parsingInstructionUber10k = """Ce document présente les équipements.
        Quand je te pose une question, renvoie moi juste la partie du document qui répond à la question, ne cherche pas à formuler autrement,
        renvoie juste la portion du document qui répond à la question"""
        parser = LlamaParse(api_key=llamaparse_api_key,
                            result_type="markdown",
                            parsing_instruction=parsingInstructionUber10k,
                            max_timeout=5000,)
        llama_parse_documents = parser.load_data(file_path)

        # Enregistrer les données analysées dans un fichier
        print("Sauvegarde des résultats de l'analyse en format .pkl...")
        joblib.dump(llama_parse_documents, data_file)

        # Définir les données analysées comme variable
        parsed_data = llama_parse_documents

    return parsed_data

@st.cache(allow_output_mutation=True)
def create_vector_database(llama_parse_documents):
    # Écrire les documents dans un fichier Markdown
    with open('data/output.md', 'a') as f:
        for doc in llama_parse_documents:
            f.write(doc.text + '\n')

    markdown_path = "/data/output.md"
    loader = UnstructuredMarkdownLoader(markdown_path)
    documents = loader.load()

    # Diviser les documents chargés en morceaux
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Initialiser les embeddings
    embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    # Créer et persister une base de données de vecteurs Chroma à partir des documents
    vs = Chroma.from_documents(
        documents=docs,
        embedding=embed_model,
        persist_directory="chroma_db_llamaparse1",
        collection_name="rag"
    )

    print('Base de données de vecteurs créée avec succès !')
    return vs, embed_model

@st.cache(allow_output_mutation=True)
def set_custom_prompt():
    prompt_template = """Utilisez les informations suivantes pour répondre à la question de l'utilisateur.
Si vous ne connaissez pas la réponse, contentez-vous de dire que vous ne savez pas, ne cherchez pas à inventer une réponse.

Contexte : {context}
Question : {question}

Ne renvoyez que la réponse utile ci-dessous et rien d'autre.
Réponse utile :
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    return prompt

# Interface utilisateur Streamlit
def main():
    st.title("Assistant de questions-réponses")

    # Fonction d'upload du fichier
    file = st.file_uploader("Téléchargez le fichier à analyser", type=["docx", "pdf"])

    if file:
        if file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":  # DOCX
            docx_text = ""
            try:
                docx_text = extract_text_from_docx(file)
            except Exception as e:
                st.error("Une erreur s'est produite lors de l'extraction du texte du fichier DOCX.")

            if docx_text:
                llama_parse_documents = load_or_parse_data(docx_text)
                vector_store, embed_model = create_vector_database(llama_parse_documents)
        elif file.type == "application/pdf":  # PDF
            pdf_text = ""
            try:
                pdf_text = extract_text_from_pdf(file)
            except Exception as e:
                st.error("Une erreur s'est produite lors de l'extraction du texte du fichier PDF.")

            if pdf_text:
                llama_parse_documents = load_or_parse_data(pdf_text)
                vector_store, embed_model = create_vector_database(llama_parse_documents)

        # Création du modèle de chat
        chat_model = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768", api_key=groq_api_key)

        # Configuration du modèle de réponse
        prompt = set_custom_prompt()

        # Création de l'objet de réponse à la requête
        qa = RetrievalQA.from_chain_type(llm=chat_model,
                                         chain_type="stuff",
                                         retriever=vector_store.as_retriever(search_kwargs={'k': 3}),
                                         return_source_documents=True,
                                         chain_type_kwargs={"prompt": prompt})

        query = st.text_input("Posez une question :", "")

        if st.button("Envoyer"):
            if query:
                # Réponse à la question posée
                response = qa.invoke({"query": query})
                if response:
                    st.write("Réponse trouvée :")
                    st.write(response)
                else:
                    st.write("Aucune réponse trouvée.")

def extract_text_from_docx(file):
    document = Document(file)
    full_text = ""
    for paragraph in document.paragraphs:
        full_text += paragraph.text + "\n"
    return full_text

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfFileReader(file)
    full_text = ""
    for page_num in range(reader.numPages):
        page = reader.getPage(page_num)
        full_text += page.extractText()
    return full_text

if __name__ == "__main__":
    main()
