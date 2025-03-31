from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()


def init_embedding_model():
    embedding = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL"))
    return embedding


# load pdfs from directory as langchain documents
def load_pdf_docs(pdf_directory_path):
    """
    Load PDF documents from the specified directory
    Returns:
    Langchain Document objects.
    """
    # Initialize PDF loader with specified directory
    loader = PyPDFDirectoryLoader(pdf_directory_path)
    # Load PDF files as langchain docs
    loaded_docs = loader.load()
    return loaded_docs


# recursively split and chunk langchain documents
def rec_split_chunk(documents, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunked_docs = text_splitter.split_documents(documents)
    return chunked_docs
