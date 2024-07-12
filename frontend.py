import speech_recognition as sr
import streamlit as st
from dotenv import load_dotenv
from template import css, bot_template, user_template
import os
import sys
from backend import (
    get_individual_pdf_path,
    get_pdf_text_from_path,
    get_text_chunks,
    get_vectorstore,
    save_embeddings_locally,
    load_embeddings_locally,
    get_conversation_chain,
    VECTOR_STORE_PATH
)

# Add current directory to PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def handle_userinput(user_question):
    if st.session_state.conversation is not None:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Speak now...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            st.write("You said: {}".format(text))
            return text
        except sr.UnknownValueError:
            st.write("Could you please Say that Again🤔")
        except sr.RequestError as e:
            st.write("Could not request results; {0}".format(e))
    return text

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Question Answering :books:")

    button_style ='''
        <style>
            .stButton > button {
            color: white;
            font-size:10px;
            background: green;
            width: 50px;
            height: 50px;
            }
        <style>
    '''

    user_question = st.text_input("Ask a question:")
    st.markdown(button_style, unsafe_allow_html=True)
    if st.button("🎙"):
        speech_text = recognize_speech()
        user_question = speech_text
    if user_question:
        handle_userinput(user_question) 

    pdf_path = "documents"
    pdf_file = get_individual_pdf_path(pdf_path)
    if pdf_file and os.path.exists(pdf_file):
        with st.spinner("Processing"):
            if os.path.exists(VECTOR_STORE_PATH):
                vectorstore = load_embeddings_locally(VECTOR_STORE_PATH)
            else:
                raw_text = get_pdf_text_from_path([pdf_file])
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                save_embeddings_locally(vectorstore, VECTOR_STORE_PATH)

            st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()