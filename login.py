import streamlit as st
from streamlit_login_auth_ui.widgets import __login__
import pandas as pd
from io import StringIO
import boto3
import os
import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfFileReader
import docx
import pandas as pd
from htbuilder import HtmlElement, div, br, hr, a, p, img, styles
from htbuilder.units import percent, px
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain import HuggingFaceHub
from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from dominate import tags
from langchain.embeddings.huggingface import HuggingFaceEmbeddings


st.cache_data.clear()
auth_token = os.environ.get("auth_token")

__login__obj = __login__(auth_token = auth_token, 
                    company_name = "Evaluatingcodes",
                    width = 200, height = 250, 
                    logout_button_name = 'Logout', hide_menu_bool = False, 
                    hide_footer_bool = False, 
                    lottie_url = 'https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json'
                    )
                    
LOGGED_IN = __login__obj.build_login_ui()

if LOGGED_IN == True:
    
    def image(src_as_string, **style):
       return img(src=src_as_string, style=styles(**style))

    def link(link, text, **style):
       return a(_href=link, _target="_blank", style=styles(**style))

    def layout(*args):

        style = """
        <style>
            # MainMenu {visibility: display;}
            footer {visibility: display;}
            .stApp { bottom: 105px; }
        </style>
        """

        style_div = """
        <style>
            position: fixed;
            left: 0;
            bottom: 0;
            margin: 0 50px;
            width: 100%;
            color: black;
            text-align: left;
            height: auto;
            opacity: 1;
        </style>
        """

        style_hr = """
        <style>
            display: block;
            margin: 8px 8px auto auto;
            border-style: inset;
            border-width: 1.5px;
        </style>
        """

        body = p()
        foot = div(style=style_div)(
            hr(style=style_hr),
            body
        )

        st.markdown(style, unsafe_allow_html=True)

        for arg in args:
            if isinstance(arg, str):
                body(arg)

            elif isinstance(arg, tags.HtmlElement):
                body(arg)

        st.markdown(str(foot), unsafe_allow_html=True)

    def main():
        #st.set_page_config(page_title="FileBot Chatify")
        st.write("# FileBot Chatify")
        st.header("Chat with Your Files")
        
        
        openai_api_key = st.sidebar.text_input('Enter your OpenAI API Key')
        if not openai_api_key:
            st.sidebar.warning("Please provide your OpenAI API key.")
            st.stop()

        load_dotenv()
        if "conversation" not in st.session_state:
            st.session_state.conversation = None
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = None
        if "processComplete" not in st.session_state:
            st.session_state.processComplete = None
        
        uploaded_files = st.file_uploader("Upload the file", type=['pdf', 'docx', 'xlsx', 'xls', 'csv', 'txt','pptx'], accept_multiple_files=True, key="file_uploader")
        # openai_api_key = os.getenv("OPENAI_API_KEY")
        # if not openai_api_key:
        #     st.warning("Please set the OPENAI_API_KEY environment variable.")
        #     st.stop()

        process = st.button("Process")

        if process:
            files_text = get_files_text(uploaded_files)
            text_chunks = get_text_chunks(files_text)
            vector_store = get_vector_store(text_chunks)
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
        pdf_reader = PdfFileReader(pdf)
        text = ""
        for page_num in range(pdf_reader.numPages):
            page = pdf_reader.getPage(page_num)
            text += page.extractText()
        return text

    def get_docx_text(file):
        doc = docx.Document(file)
        all_text = [doc_para.text for doc_para in doc.paragraphs]
        return ' '.join(all_text)

    def get_excel_text(file):
        try:
            df = pd.read_excel(file, sheet_name=0)
            text = ' '.join(df.applymap(str).values.flatten())
            return text
        except Exception as e:
            st.warning(f"Error reading Excel file: {e}")
            return ""

    def get_csv_text(file):
        try:
            df = pd.read_csv(file)
            text = ' '.join(df['text'].astype(str).values)
            return text
        except Exception as e:
            st.warning(f"Error reading CSV file: {e}")
            return ""

    def get_txt_text(file):
        try:
            with open(file, 'r', encoding='utf-8') as txt_file:
                text = txt_file.read()
            return text
        except Exception as e:
            st.warning(f"Error reading text file: {e}")
            return ""

    def get_text_chunks(text):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=900,
            chunk_overlap=100,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks

    # def get_vector_store(text_chunks):
    #     embeddings_model = HuggingFaceEmbeddings(model_name="bert-base-uncased")

    #     embeddings = []
    #     for chunk in text_chunks:
    #         embedding = embeddings_model.encode([chunk])[0]
    #         embeddings.append(embedding)

    #     vector_store = FAISS.from_vectors(embeddings)

    #     return vector_store

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

        response_container = st.container()

        with response_container:
            for i, messages in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    message(messages.content, is_user=True, key=str(i))
                else:
                    message(messages.content, key=str(i))
        st.markdown('''
        Made by Eva Saini,
        [LinkedIN](https://www.linkedin.com/in/eva-saini-b0909a24a)
        ''')
        #st.markdown(custom_css, unsafe_allow_html=True)             

    if __name__ == '__main__':
       main()
