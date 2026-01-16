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
    st.error("❌ GROQ_API_KEY not found.")
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
    return_messages=True,
    output_key="answer"
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

# ───────────────── Styling ─────────────────
st.markdown("""
<style>

/* ===== Global Background ===== */
.stApp {
    background-image:st.get_media_file_url("ai-digital.gif");
    background-repeat: no-repeat;
    background-position: center;
    background-size: cover;
}

/* ===== Title ===== */
h1 {
    color: white !important;
    text-shadow: 1px 1px 4px black;
}

/* Prevent overlap with input */
.main .block-container {
    padding-bottom: 150px !important;
}

/* ===== Chat Layout ===== */
.chat-row {
    display: flex;
    margin: 0.7rem 0;
    align-items: flex-end;
}

.chat-row.bot { justify-content: flex-start; }
.chat-row.user { justify-content: flex-end; }

.chat-avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    margin: 0 8px;
}

/* ===== Bubbles ===== */
.bot-bubble,
.user-bubble {
    background: rgba(10,10,10,0.90);
    color: white;
    padding: 12px 16px;
    border-radius: 16px;
    max-width: 70%;
    word-wrap: break-word;
}

.bot-bubble { border-bottom-left-radius: 4px; }
.user-bubble { border-bottom-right-radius: 4px; }

/* ===== Buttons ===== */
div[data-testid="stButton"] > button {
    background: rgba(10,10,10,0.9);
    color: white;
    border-radius: 16px;
    padding: 10px 14px;
    border: 1px solid #333;
}

/* ===== Input ===== */
input[type="text"] {
    background: rgba(10,10,10,0.95) !important;
    color: white !important;
    border-radius: 16px !important;
    border: 1px solid #333 !important;
}

/* ===== Mobile Fix ===== */
@media (max-width: 768px) {

    .bot-bubble,
    .user-bubble {
        max-width: 88%;
        font-size: 14px;
        padding: 10px 12px;
    }

    .chat-avatar {
        width: 26px;
        height: 26px;
        margin: 0 6px;
    }

    .main .block-container {
        padding-bottom: 180px !important;
    }

    * {
        backdrop-filter: none !important;
        -webkit-backdrop-filter: none !important;
    }
}

</style>
""", unsafe_allow_html=True)

# ───────────────── Header ─────────────────
st.title("✨ Ayesha's Career Chatbot")
st.markdown(
    "Ask anything about **education, skills, experience & projects**"
)

BOT_AVATAR = st.get_media_file_url("bot.png")
USER_AVATAR = st.get_media_file_url("user.png")

# ───────────────── Send Message ─────────────────
def send_message(text):
    st.session_state.messages.append(
        {"role": "user", "content": text}
    )

    with st.spinner("Thinking..."):
        result = qa_chain.invoke({"question": text})

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"]}
    )

# ───────────────── Suggestions ─────────────────
if len(st.session_state.messages) == 0:
    cols = st.columns(3)
    suggestions = [
        "Tell me about Ayesha's projects",
        "What internships has Ayesha done?",
        "What are Ayesha's strongest skills?"
    ]
    for col, text in zip(cols, suggestions):
        if col.button(text):
            send_message(text)
            st.rerun()

# ───────────────── Chat Messages ─────────────────
for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        st.markdown(f"""
        <div class="chat-row bot">
            <img src="{BOT_AVATAR}" class="chat-avatar">
            <div class="bot-bubble">{msg['content']}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-row user">
            <div class="user-bubble">{msg['content']}</div>
            <img src="{USER_AVATAR}" class="chat-avatar">
        </div>
        """, unsafe_allow_html=True)

# ───────────────── Chat Input ─────────────────
if prompt := st.chat_input("Ask anything about Ayesha's CV..."):
    if prompt.strip():
        send_message(prompt)
        st.rerun()
