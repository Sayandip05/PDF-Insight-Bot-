import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load API keys
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Streamlit title
st.title("📄 Gemma Model Document Q&A")

# Load LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="Llama3-8b-8192"
)

# Prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# Vector embedding function
def vector_embedding():
    if "vectors" not in st.session_state:
        try:
            st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            st.session_state.loader = PyPDFDirectoryLoader("./pdf_file")  # ✅ Use 'pdf_file' folder
            st.session_state.docs = st.session_state.loader.load()

            if not st.session_state.docs:
                st.error("❌ No PDF documents found in ./pdf_file folder.")
                return

            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(
                st.session_state.docs[:20]
            )

            if not st.session_state.final_documents:
                st.error("❌ Document splitting failed. No content to vectorize.")
                return

            st.session_state.vectors = FAISS.from_documents(
                st.session_state.final_documents,
                st.session_state.embeddings
            )
            st.success("✅ Vector Store DB is ready.")
        except Exception as e:
            st.error(f"⚠️ Embedding failed: {e}")

# UI input
prompt1 = st.text_input("💬 Enter your question from the documents:")

# Embed documents on button click
if st.button("📂 Embed Documents"):
    vector_embedding()

# If a question is asked
if prompt1:
    if "vectors" not in st.session_state:
        st.error("⚠️ Please embed the documents first using the button above.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        duration = time.process_time() - start
        print("Response time:", duration)

        st.write("### 📌 Answer:")
        st.write(response['answer'])

        # Show document chunks used
        with st.expander("📚 Document Similarity Search (Context Chunks)"):
            for i, doc in enumerate(response.get("context", [])):
                st.write(doc.page_content)
                st.write("--------")
