from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
import os
import argparse
import embedding

collection_name = os.getenv("CHROMA_COLLECTION_NAME")
embedding_model = embedding.init_embedding_model()
persist_directory = os.getenv("CHROMA_PERSIST_PATH")
chunk_size = os.getenv("CHUNK_SIZE")
chunk_overlap = os.getenv("OVERLAP")


def chroma_unique_id(data):
    data = data.to_pandas()
    data["unique_id"] = data.index
    print("indexing complete")


def query_database(collection, query_text, n_results=5):
    results = collection.query(query_texts=query_text, n_results=n_results)
    return results


def upload_to_collection(
    collection_name, chunked_documents, embedding_model, persist_path=None
):

    if persist_path:
        vectordb = Chroma.from_documents(
            collection_name=collection_name,
            documents=chunked_documents,
            embedding=embedding_model,
            persist_directory=persist_path,
        )

    else:
        vectordb = Chroma.from_documents(
            collection_name=collection_name,
            documents=chunked_documents,
            embedding=embedding_model,
        )
    return vectordb


def get_vector_db(collection_name, embedding_model, persist_path):
    db = Chroma(
        collection_name=collection_name,
        persist_directory=persist_path,
        embedding_function=embedding_model,
    )
    return db


if __name__ == "__main__":
    # create chroma db of documents
    # user params
    parser = argparse.ArgumentParser(
        prog="upload docs to chromadb",
        description="Load files in directory to chroma vector db",
    )

    parser.add_argument("directory", type=str, help="location of file directory")
    args = parser.parse_args()

    # load docs as langchain docs
    print("loading docs as langchain documents")
    loader = DirectoryLoader(
        args.directory, use_multithreading=True, show_progress=True
    )  # glob="**/*.md"
    docs = loader.load()
    print(f"loaded {len(docs)} docs")

    # split and chunk the documents to prep for embedding
    print("splitting and chunking documents")
    chunk_size = int(os.getenv("CHUNK_SIZE", 3000))
    chunk_overlap = int(os.getenv("OVERLAP", 500))
    print(f"Using chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    chunked = embedding.rec_split_chunk(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # create persistent vector db of embeddings in chroma
    print(f"uploading documents to chroma collection: {collection_name}")
    batch_size = 41000

    def batch_process(chunked, batch_size):
        for i in range(0, len(chunked), batch_size):
            batch = chunked[i : i + batch_size]
            print(f"uploading chunked docs batch {i} - {i+batch_size}")
            upload_to_collection(
                collection_name=collection_name,
                chunked_documents=batch,
                embedding_model=embedding_model,
                persist_path=persist_directory,
            )

    batch_process(chunked, batch_size)
    print("complete")
