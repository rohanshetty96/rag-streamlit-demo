import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_mongodb import MongoDBAtlasVectorSearch
import tempfile

load_dotenv()
embeddings = OpenAIEmbeddings(openai_api_key=os.environ["LLM_API_KEY"])

st.set_page_config(page_title="RAG PDF Chat", layout="centered")
st.title("📘 RAG Chat: Ask Anything")

st.markdown("Upload your own PDF or get to know more about AI security.")

pdf_file = st.file_uploader("📄 Upload PDF", type="pdf")
query = st.text_input("🔍 Enter your question:")

retriever = None

if pdf_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(pages)

    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=None)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    source_type = "User-uploaded PDF"
else:
    vectorstore = MongoDBAtlasVectorSearch.from_connection_string(
        os.environ["MONGODB_URI"],
        "book_mongodb_chunks.rag_documents",
        embeddings,
        index_name="vector_index"
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10, "pre_filter": {"hasCode": False}, "score_threshold": 0.7}
    )
    source_type = "Preloaded MongoDB PDF"

# Setup RAG pipeline
template = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say so. Do not make anything up.

Context:
{context}

Question: {question}
"""
prompt = PromptTemplate.from_template(template)
llm = ChatOpenAI(openai_api_key=os.environ["LLM_API_KEY"], temperature=0)

rag_chain = {
    "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
    "question": RunnablePassthrough()
} | prompt | llm | StrOutputParser()

# Run RAG
if query:
    with st.spinner(f"🧠 Answering using: {source_type}..."):
        docs = retriever.invoke(query)
        answer = rag_chain.invoke(query)

    st.markdown("### 💬 Answer")
    st.write(answer)

    # 🛑 Only show chunks if the model didn't say "I don't know"
    if "don't know" not in answer.lower():
        st.markdown("### 🔍 Top Matching Chunks")
        for i, doc in enumerate(docs):
            st.markdown(f"**Chunk {i+1}**")
            st.info(doc.page_content[:1000])
    else:
        st.info("No relevant information found in the document.")