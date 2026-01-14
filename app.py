import streamlit as st
import os
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Ayesha's Career Chatbot",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Environment ──────────────────────────────────────────────
load_dotenv()
if "GROQ_API_KEY" not in os.environ:
    st.error("❌ GROQ_API_KEY not found. Please add it to your .env file.")
    st.stop()

CV_PATH = "cv.pdf"
INDEX_DIR = "chroma_index"

# ── Vector Store ─────────────────────────────────────────────
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(INDEX_DIR):
        return Chroma(persist_directory=INDEX_DIR, embedding_function=embeddings)

    loader = PyPDFLoader(CV_PATH) if CV_PATH.lower().endswith(".pdf") else TextLoader(CV_PATH, encoding="utf8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=INDEX_DIR)
    return vectorstore

# ── AI Setup ─────────────────────────────────────────────────
vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

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
    retriever=retriever,
    memory=memory,
    return_source_documents=False
)

# ── Session State ────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Styling ──────────────────────────────────────────────────
st.markdown("""
<style>
.stApp {
    background: url("https://c.tenor.com/Ho0ZextTZJEAAAAC/ai-digital.gif")
    no-repeat center center fixed;
    background-size: cover;
}

h1 {
    color: white !important;
    text-shadow: 1px 1px 4px #000;
}

/* Main content area - add padding at bottom for fixed input */
.main .block-container {
    padding-bottom: 120px !important;
}



/* Chat layout */
.chat-row {
    display: flex;
    margin: 0.8rem 0;
    align-items: flex-end;
}

.chat-row.bot { justify-content: flex-start; }
.chat-row.user { justify-content: flex-end; }

.chat-avatar {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    margin: 0 10px;
    border: 2px solid #222;
}

/* Bot bubble */
.bot-bubble {
    background: rgba(10,10,10,0.90);
    color: white;
    border-radius: 16px;
    padding: 12px 16px;
    max-width: 68%;
    word-wrap: break-word;
    animation: fadeIn 0.4s ease-out;
    border-bottom-left-radius: 4px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.6);
}

/* User bubble */
.user-bubble {
    background: rgba(10,10,10,0.90);
    color: white;
    border-radius: 16px;
    padding: 12px 16px;
    max-width: 68%;
    border-bottom-right-radius: 4px;
}

/* Suggestions */
.suggestion-row {
    display: flex;
    justify-content: center;
    gap: 12px;
    margin: 1rem 0;
    flex-wrap: wrap;
}

div[data-testid="stButton"] > button {
    background: rgba(10,10,10,0.90);
    color: white;
    border-radius: 16px;
    padding: 12px 16px;
    border: 1px solid #333;
    font-size: 14px;
    max-width: 260px;
    text-align: left;
    box-shadow: 0 6px 18px rgba(0,0,0,0.6);
    transition: all 0.2s ease;
}

div[data-testid="stButton"] > button:hover {
    background: rgba(20,20,20,0.95);
    transform: translateY(-2px);
}

/* Fixed input container at bottom */
.fixed-input-container {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: rgba(10,10,10,0.95);
    padding: 16px;
    box-shadow: 0 -4px 12px rgba(0,0,0,0.5);
    z-index: 999;
    backdrop-filter: blur(10px);
}

/* Chat input */
input[type="text"] {
    background: rgba(10,10,10,0.9) !important;
    color: white !important;
    border-radius: 16px !important;
    border: 1px solid #333 !important;
    padding: 10px 14px !important;
    width: 100%;
}

input[type="text"]::placeholder {
    color: #aaa !important;
}

/* Fade animation */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────
st.title("✨ Ayesha's Career Chatbot")
st.markdown("Ask anything about **education, skills, experience & projects** — powered by Ayesha's CV")

BOT_AVATAR = "https://cdn-icons-png.flaticon.com/512/4712/4712107.png"
USER_AVATAR = "https://cdn-icons-png.flaticon.com/512/1077/1077063.png"

# ── Function to handle message sending ──────────────────────
def send_message(text):
    st.session_state.messages.append({"role": "user", "content": text})
    with st.spinner("Thinking..."):
        result = qa_chain.invoke({"question": text})
    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})


# ── Suggestions ──────────────────────────────────────────────
if len(st.session_state.messages) == 0:
    st.markdown('<div class="suggestion-row">', unsafe_allow_html=True)

    suggestions = [
        "Tell me about Ayesha's projects",
        "What internships has Ayesha done?",
        "What are Ayesha's strongest technical skills?"
    ]

    cols = st.columns(3)
    for col, text in zip(cols, suggestions):
        if col.button(text):
            send_message(text)

    st.markdown('</div>', unsafe_allow_html=True)

# ── Chat Messages ────────────────────────────────────────────
st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)

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

st.markdown('</div>', unsafe_allow_html=True)

# ── Chat Input (Fixed at bottom) ─────────────────────────────
prompt = st.chat_input("Ask anything about Ayesha's CV...")
if prompt:
    send_message(prompt)
