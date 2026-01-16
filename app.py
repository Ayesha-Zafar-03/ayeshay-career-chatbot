import streamlit as st
import os
from dotenv import load_dotenv

# LangChain
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq

# ───────────────── Page Config ─────────────────
st.set_page_config(
    page_title="Ayesha's Career Chatbot",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ───────────────── Env ─────────────────
load_dotenv()
if "GROQ_API_KEY" not in os.environ:
    st.error("❌ GROQ_API_KEY not found")
    st.stop()

CV_PATH = "cv.pdf"
INDEX_DIR = "chroma_index"

# ───────────────── Vector Store ─────────────────
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if os.path.exists(INDEX_DIR):
        return Chroma(
            persist_directory=INDEX_DIR,
            embedding_function=embeddings
        )

    loader = (
        PyPDFLoader(CV_PATH)
        if CV_PATH.lower().endswith(".pdf")
        else TextLoader(CV_PATH, encoding="utf8")
    )

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    return Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=INDEX_DIR
    )

vectorstore = load_vectorstore()

# ───────────────── AI Setup ─────────────────
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 12}),
    memory=memory,
    return_source_documents=False
)

# ───────────────── Session State ─────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ───────────────── Adaptive Styling ─────────────────
st.markdown("""
<style>
/* Desktop background */
@media (min-width: 769px) {
    .stApp {
        background-image: url("https://c.tenor.com/Ho0ZextTZJEAAAAC/ai-digital.gif");
        background-size: cover;
        background-position: center;
    }
}

/* Mobile fallback */
@media (max-width: 768px) {
    .stApp {
        background-color: #0f0f0f;
    }
}

h1 {
    color: white;
    text-shadow: 1px 1px 4px black;
}
</style>
""", unsafe_allow_html=True)

# ───────────────── Header ─────────────────
st.title("✨ Ayesha's Career Chatbot")
st.caption("Ask anything about education, skills, experience & projects")

# ───────────────── CV Download (SAFE) ─────────────────
with open(CV_PATH, "rb") as f:
    st.download_button(
        label="📄 Download CV",
        data=f,
        file_name="Ayesha_Zafar_CV.pdf",
        mime="application/pdf"
    )

st.divider()

# ───────────────── Chat History ─────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ───────────────── Chat Input ─────────────────
if prompt := st.chat_input("Ask anything about Ayesha's profile…"):
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = qa_chain.invoke({"question": prompt})
            answer = response["answer"]
            st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
