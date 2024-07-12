import os
import glob
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from sklearn.metrics.pairwise import cosine_similarity

VECTOR_STORE_PATH = "embeddings.npy"
pdf_path = "documents"

def get_pdf_files(pdf_path):
    pdf_path = os.path.abspath(pdf_path)
    pdf_files = glob.glob(os.path.join(pdf_path, "*.pdf"))
    return pdf_files

def create_file(file_path):
    file_dir = os.path.dirname(file_path)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    with open(file_path, 'w') as f:
        pass

def get_pdf_text_from_path(pdf_files):
    text = ""
    for pdf_file in pdf_files:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def save_embeddings_locally(vectorstore, file_path):
    vectorstore.save_local(file_path)

def load_embeddings_locally(file_path):
    return FAISS.load_local(file_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def get_individual_pdf_path(pdf_path):
    pdf_files = get_pdf_files(pdf_path)
    return pdf_files[0] if pdf_files else None