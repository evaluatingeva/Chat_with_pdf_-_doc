import streamlit as st
import os
from PyPDF2 import PdfReader
import docx
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain import HuggingFaceHub
from streamlit_chat import message
from langchain.callbacks import get_openai_callback

# Custom CSS
custom_css = """
<style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f5f5f5;
        padding: 20px;
    }
    .container {
        max-width: 800px;
        margin: auto;
        padding: 20px;
        background-color: #fff;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color: #333;
    }
    button {
        background-color: #4caf50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    .file-uploader {
        margin-bottom: 20px;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

def main():
    load_dotenv()
    st.write("Before set_page_config")
    st.set_page_config(page_title="FileBot Chatify")
    st.write("After set_page_config")
    
    st.header("💬 Chat Filebot")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    uploaded_files = st.file_uploader("Upload the file", type=['pdf', 'docx', 'xlsx', 'xls', 'csv', 'txt'], accept_multiple_files=True, key="file_uploader")
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    process = st.button("Process")

    if process:
        if not openai_api_key:
            st.info("Add your OPENAI_Key here")
            st.stop()
        files_text = get_files_text(uploaded_files)
        # get text chunks
        text_chunks = get_text_chunks(files_text)
        # create vector stores
        vector_store = get_vector_store(text_chunks)
        # create conversation chain
        st.session_state.conversation = get_conversation_chain(vector_store, openai_api_key)
        st.session_state.processComplete = True

    if st.session_state.processComplete:
        user_question = st.text_input("Ask Question ")
        if user_question:
            handle_user_input(user_question)

def get_files_text(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        split_tup = os.path.splitext(uploaded_file.name)
        file_extension = split_tup[1]
        if file_extension == ".pdf":
            text += get_pdf_text(uploaded_file)
        elif file_extension == ".docx":
            text += get_docx_text(uploaded_file)
        elif file_extension == ".xlsx" or file_extension == ".xls":
            text += get_excel_text(uploaded_file)
        elif file_extension == ".csv":
            text += get_csv_text(uploaded_file)
        elif file_extension == ".txt":
            text += get_txt_text(uploaded_file)
        else:
            # Handle other file types if needed
            pass
    return text

def get_pdf_text(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_docx_text(file):
    doc = docx.Document(file)
    all_text = [doc_para.text for doc_para in doc.paragraphs]
    return ' '.join(all_text)

def get_excel_text(file):
    try:
        # Assuming the first sheet in the Excel file contains text data
        df = pd.read_excel(file, sheet_name=0)
        text = ' '.join(df.applymap(str).values.flatten())
        return text
    except Exception as e:
        # Handle exceptions, e.g., if the file is not a valid Excel file
        print(f"Error reading Excel file: {e}")
        return ""

def get_csv_text(file):
    try:
        # Assuming the CSV file has a column named 'text' containing text data
        df = pd.read_csv(file)
        text = ' '.join(df['text'].astype(str).values)
        return text
    except Exception as e:
        # Handle exceptions, e.g., if the file is not a valid CSV file
        print(f"Error reading CSV file: {e}")
        return ""

def get_txt_text(file):
    try:
        with open(file, 'r', encoding='utf-8') as txt_file:
            text = txt_file.read()
        return text
    except Exception as e:
        # Handle exceptions, e.g., if the file is not a valid text file
        print(f"Error reading text file: {e}")
        return ""

def get_text_chunks(text):
    # split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings()
    knowledge_base = FAISS.from_texts(text_chunks, embeddings)
    return knowledge_base

def get_conversation_chain(vector_store, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_user_input(user_question):
    with get_openai_callback() as cb:
        response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Layout of input/response containers
    response_container = st.container()

    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                message(messages.content, is_user=True, key=str(i))
            else:
                message(messages.content, key=str(i))
st.markdown('''
Made by Eva Saini,
link("https://www.linkedin.com/in/eva-saini-b0909a24a", "@Eva")
''')                
                

if __name__ == '__main__':
    main()
