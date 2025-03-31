from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.llms import OllamaLLM
from operator import itemgetter
import os

from dotenv import load_dotenv

load_dotenv()

chat_history_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a disaster response assistant with expertise in FEMA guidelines and procedures. Your role is to provide accurate information from FEMA documents to help people understand disaster preparedness, response, and recovery procedures.

Answer the question based ONLY on the following context:
{context}

Guidelines:
1. Use ONLY the information provided in the FEMA documentation context.
2. Be specific about FEMA procedures, requirements, and guidelines.
3. When referencing information, always cite the specific FEMA document, section, and page number.
4. If asked about eligibility criteria, funding, or deadlines, be very precise with the requirements and conditions.
5. For procedural questions, present the steps in a clear, numbered format.
6. If the question requires information not present in the context, clearly state that you need to refer to additional FEMA documentation.
7. If you're unsure about any details, err on the side of caution and recommend consulting official FEMA resources.

Remember: Your responses can impact disaster preparedness and recovery efforts, so accuracy is critical."""
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

def format_docs(docs):
    formatted_docs = []
    for i, doc in enumerate(docs):
        # Extract metadata if available
        metadata = getattr(doc, 'metadata', {})
        source = metadata.get('source', 'Unknown source')
        page = metadata.get('page', 'Unknown page')
        
        # Format the document with metadata
        formatted_doc = f"[Document {i+1}] "
        if 'page' in metadata:
            formatted_doc += f"Page {page}: "
        formatted_doc += doc.page_content
        
        formatted_docs.append(formatted_doc)
    
    return "\n\n" + "-" * 50 + "\n\n".join(formatted_docs) + "\n\n" + "-" * 50


def init_llm():
    llm = OllamaLLM(model=os.getenv("LLM"))
    return llm


def query_rag_streamlit(Chroma_collection, llm_model, promp_template):
    """
    Query a Retrieval-Augmented Generation (RAG) system using Chroma db.
    Args:
      - query_text (str): The text to query the RAG system with.
      - prompt_template (str): Query prompt template
      inclding context and question
    Returns:
      - formatted_response (str): Formatted response including
      the generated text and sources.
      - response_text (str): The generated response text.
    """

    # Use the global format_docs function

    db = Chroma_collection

    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.3, "k": 8},
    )

    context = itemgetter("question") | retriever | format_docs
    first_step = RunnablePassthrough.assign(context=context)
    chain = first_step | promp_template | llm_model

    return chain
