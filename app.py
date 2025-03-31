import streamlit as st
from langchain_chroma import Chroma
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)
from langchain_community.llms import Ollama
import os

from util import embedding
from util import query


chroma_collection_name = os.getenv("CHROMA_COLLECTION_NAME")
embedding_model = embedding.init_embedding_model()
chroma_persist_path = os.getenv("CHROMA_PERSIST_PATH")
llm = os.getenv("LLM")

# check if docker is running
def is_docker():
    def text_in_file(text, filename):
        try:
            with open(filename, encoding="utf-8") as lines:
                return any(text in line for line in lines)
        except OSError:
            return False

    cgroup = "/proc/self/cgroup"
    return os.path.exists("/.dockerenv") or text_in_file("docker", cgroup)


if not is_docker():
    llm = query.init_llm()
else:
    # for ollama in docker image
    llm = Ollama(model=llm, base_url="http://ollama-container:11434")

# Load data from vector db
db = Chroma(
    collection_name=chroma_collection_name,
    embedding_function=embedding_model,
    persist_directory=chroma_persist_path,
    collection_metadata={"hnsw:space": "cosine"},
)

# Setting the title of the Streamlit application
st.title("Ollama and Chroma Langchain RAG")


msgs = StreamlitChatMessageHistory(key="special_app_key")
history = StreamlitChatMessageHistory(key="chat_messages")


if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

template = query.chat_history_template

chain = query.query_rag_streamlit(db, llm, template)
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,  # Always return the instance created earlier
    input_messages_key="question",
    history_messages_key="history",
)

for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if prompt := st.chat_input():
    st.chat_message("human").write(prompt)

    # As usual, new messages are added to StreamlitChatMessageHistory
    # when the Chain is called.
    config = {"configurable": {"session_id": "any"}}
    response = chain_with_history.invoke({"question": prompt}, config)
    st.chat_message("ai").write(response)
