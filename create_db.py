from langchain.document_loaders import UnstructuredFileLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
import os


# 加载文件函数
def get_persist_vectordb(file_path, model_name, persist_directory):

    loader = PyPDFLoader(file_path)
    docs = loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100))
    docs = [doc for doc in docs if doc.metadata['page']>18]

    embeddings = HuggingFaceEmbeddings(model_name=model_name
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory  
    )
    vectordb.persist()


if __name__ == '__main__':
    file_path = "InternLM/data/哲学小辞典合集（毛主义哲学再版）.pdf"
    model_name ="D:/Python/NLP工程/model_weights/bge-base-zh"
    persist_directory = 'InternLM/data_base/vector_db/chroma'
    get_persist_vectordb(file_path, model_name, persist_directory)