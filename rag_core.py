import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

def get_mongo_retriever():
    vectorstore = MongoDBAtlasVectorSearch.from_connection_string(
        os.environ["MONGODB_URI"],
        "book_mongodb_chunks.rag_documents",
        OpenAIEmbeddings(openai_api_key=os.environ["LLM_API_KEY"]),
        index_name="vector_index"
    )
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 10,
            "pre_filter": { "hasCode": False },
            "score_threshold": 0.01
        }
    )

def run_rag_pipeline(query: str):
    retriever = get_mongo_retriever()
    docs = retriever.invoke(query)

    # Build prompt
    template = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know.
    
    Context:
    {context}
    
    Question: {question}
    """
    prompt = PromptTemplate.from_template(template)
    llm = ChatOpenAI(openai_api_key=os.environ["LLM_API_KEY"], temperature=0)
    rag_chain = {
        "context": lambda q: "\n\n".join([d.page_content for d in docs]),
        "question": RunnablePassthrough()
    } | prompt | llm | StrOutputParser()

    answer = rag_chain.invoke(query)
    return answer, docs
