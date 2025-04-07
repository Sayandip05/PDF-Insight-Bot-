# 🤖 PDF Insight Bot

**PDF Insight Bot** is an AI-powered assistant that lets you upload PDF documents and ask questions in plain English. It retrieves context-aware answers using powerful language models and vector search—built for students, researchers, professionals, and anyone working with long documents.


## 🚀 Key Features

✅ Upload and process multiple PDFs  
✅ Ask questions in natural language  
✅ Context-based answers using **Groq's Llama3-8b**  
✅ Uses **Google Generative AI embeddings** for document vectorization  
✅ Efficient document retrieval with **FAISS Vector Store**  
✅ Smooth and interactive UI built with **Streamlit**

---

## 🧠 Tech Stack

| Layer            | Tools & Technologies                      |
|------------------|--------------------------------------------|
| **Frontend**     | Streamlit                                  |
| **LLM**          | Groq API (Llama3-8b)                       |
| **Embeddings**   | Google Generative AI (embedding-001)       |
| **Retrieval**    | FAISS Vector Store                         |
| **PDF Parsing**  | LangChain + PyPDFDirectoryLoader           |
| **Orchestration**| LangChain Framework                        |

---

## 📂 How It Works

1. 📥 **Upload** your PDF files into the `pdf_file/` directory  
2. 🧩 **Chunks** are created from documents using recursive text splitting  
3. 🧠 **Embeddings** are generated using Google’s GenAI embedding model  
4. 📚 **Vectors** are stored using FAISS  
5. ❓ **Ask questions** via the app — answers are generated using Groq’s Llama3 model based on relevant context from your documents  
