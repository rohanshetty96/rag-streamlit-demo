from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_transformers.openai_functions import create_metadata_tagger
import key_param

# MongoDB Connection
client = MongoClient(key_param.MONGODB_URI)
dbName = "book_mongodb_chunks"
collectionName = "rag_documents"  # ✅ Use a fresh collection
collection = client[dbName][collectionName]

# OPTIONAL: Drop old collection if it exists
collection.drop()

# Load PDF
loader = PyPDFLoader("/Users/rohanshetty/Projects/RAG_Langchain/Navigating_the_security_landscape_of_generative_AI.pdf")
pages = loader.load()

# Filter short pages
cleaned_pages = [page for page in pages if len(page.page_content.split(" ")) > 20]

# Split pages into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)

# Setup metadata tagging schema
schema = {
    "properties": {
        "title": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "hasCode": {"type": "boolean"},
    },
    "required": ["title", "keywords", "hasCode"],
}

# Use OpenAI LLM for metadata tagging
llm = ChatOpenAI(
    openai_api_key=key_param.LLM_API_KEY,
    temperature=0,
    model="gpt-3.5-turbo"
)

document_transformer = create_metadata_tagger(metadata_schema=schema, llm=llm)
docs = document_transformer.transform_documents(cleaned_pages)
split_docs = text_splitter.split_documents(docs)

# Generate embeddings
embeddings = OpenAIEmbeddings(openai_api_key=key_param.LLM_API_KEY)

# Save to MongoDB Atlas with vector index
vectorStore = MongoDBAtlasVectorSearch.from_documents(
    split_docs,
    embeddings,
    collection=collection
)

print("✅ Successfully inserted documents into rag_documents collection.")



