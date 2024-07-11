import streamlit as st
from dotenv import load_dotenv
from template import css, bot_template, user_template
import os

from backend import(
    get_individual_pdf_path,
    get_pdf_text_from_path,
    get_text_chunks,
    get_vectorstore,
    save_embeddings_locally,
    load_embeddings_locally,
    get_conversation_chain,
    VECTOR_STORE_PATH
)

def handle_userinput(user_question):
    if st.session_state.conversation is not None:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask a question", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Ask a Question regarding the course:books:")
    user_question = st.text_input("Ask a question:")
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