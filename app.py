import chainlit as cl
from operator import itemgetter

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Qdrant


# Constants (you can adjust these as per your environment)
# DATA LOADER
DATA_LINK1 = "https://www.whitehouse.gov/wp-content/uploads/2022/10/Blueprint-for-an-AI-Bill-of-Rights.pdf"
DATA_LINK2 = "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf"

# CHUNKING CONFIGS
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# RETRIEVER CONFIGS
COLLECTION_NAME = "AI Bill of Rights"

EMBEDDING_MODEL = "text-embedding-3-small"

# FINAL RAG CONFIGS
QA_MODEL = "gpt-4o"

RAG_PROMPT = """\
Given a provided context and question, you must answer the question based only on context.

If you cannot answer the question based on the context - you must say "I don't know".

Context: {context}
Question: {question}
"""

# Function to chunk documents
def chunk_documents(unchunked_documents, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_documents(unchunked_documents)

# Function to build retriever
def build_retriever(chunked_documents, embeddings, collection_name):
    vectorstore = Qdrant.from_documents(
        documents=chunked_documents,
        embedding=embeddings,
        location=":memory:",  # Storing in-memory for demonstration
        collection_name=collection_name,
    )
    retriever = vectorstore.as_retriever()
    return retriever

# Load documents and prepare retriever
rag_documents_1 = PyMuPDFLoader(file_path=DATA_LINK1).load()
rag_documents_2 = PyMuPDFLoader(file_path=DATA_LINK2).load()

chunked_rag_documents = chunk_documents(rag_documents_1, CHUNK_SIZE, CHUNK_OVERLAP) + \
                        chunk_documents(rag_documents_2, CHUNK_SIZE, CHUNK_OVERLAP)

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
retriever = build_retriever(chunked_rag_documents, embeddings, COLLECTION_NAME)

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
qa_llm = ChatOpenAI(model=QA_MODEL)

rag_chain = (
    {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
    | rag_prompt | llm | StrOutputParser()
)

# Chainlit app
@cl.on_message
async def main(message: str):
    response = rag_chain.invoke({"question": message})
    await cl.Message(
        content=response["response"],  # Extract the response from the chain
        author="AI"
    ).send()
