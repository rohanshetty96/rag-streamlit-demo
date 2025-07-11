# 🧠 RAG Chat with LangChain + MongoDB + Chroma + Streamlit

This is a full-stack Retrieval-Augmented Generation (RAG) app built with:

- 📘 MongoDB Atlas Vector Search (pre-indexed PDF)
- 📄 FAISS/Chroma (user-uploaded PDF)
- 🧠 OpenAI GPT-3.5 for answers
- 🚀 Streamlit UI

## 🔧 Features

- Ask questions about a preloaded PDF from MongoDB
- Upload your own PDF and ask about it
- View top 10 matching chunks
- Hosted via Streamlit Community Cloud

## 🛠️ Local Setup

```bash
git clone https://github.com/your-username/rag-streamlit-demo.git
cd rag-streamlit-demo
pip install -r requirements.txt
```
# RAG PDF Chat with LangChain + Streamlit

📘 Ask questions over your own PDFs or a preloaded database using a RAG pipeline.

🚀 [Live Demo](https://rag-streamlit-demo.onrender.com)

---

This app supports:
- PDF uploads for custom document QA
- MongoDB Atlas vector search for fallback
- LangChain + OpenAI integration
