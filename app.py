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

# ── Environment & Safety Check ────────────────────────────────
load_dotenv()
if "GROQ_API_KEY" not in os.environ:
    st.error("❌ GROQ_API_KEY not found. Please add it to your .env file.")
    st.stop()

CV_PATH = "cv.pdf"
INDEX_DIR = "chroma_index"

# ── Vector Store (cached) ─────────────────────────────────────
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if os.path.exists(INDEX_DIR):
        return Chroma(persist_directory=INDEX_DIR, embedding_function=embeddings)
    
    loader = PyPDFLoader(CV_PATH) if CV_PATH.lower().endswith(".pdf") else TextLoader(CV_PATH, encoding="utf8")
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)
    
    vectorstore = Chroma.from_documents(
        chunks, embeddings, persist_directory=INDEX_DIR
    )
    return vectorstore

# ── Initialize AI Components ──────────────────────────────────
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

# ── Session State ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Styling ───────────────────────────────────────────────────
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
    
    .chat-wrapper {
        height: 68vh;
        overflow-y: auto;
        padding: 1rem;
        background: rgba(0,0,0,0.45);
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.08);
        backdrop-filter: blur(4px);
    }
    
    .chat-row {
        display: flex;
        margin: 0.8rem 0;
        align-items: flex-end;
    }
    
    .chat-row.bot {
        justify-content: flex-start;
    }
    
    .chat-row.user {
        justify-content: flex-end;
    }
    
    .chat-avatar {
        width: 38px;
        height: 38px;
        border-radius: 50%;
        margin: 0 10px;
        border: 2px solid #222;
    }
    
    .bot-bubble, .user-bubble {
        border-radius: 16px;
        padding: 10px 16px;
        max-width: 68%;
        word-wrap: break-word;
        animation: fadeIn 0.4s ease-out;
    }
    
    .bot-bubble {
        background: rgba(30,30,30,0.85);
        color: white;
        border-bottom-left-radius: 4px;
    }
    
    .user-bubble {
        background: #1E3A8A;
        color: white;
        border-bottom-right-radius: 4px;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# ── Header & Download ─────────────────────────────────────────
st.title("✨ Ayesha's Career Chatbot")
st.markdown("Ask anything about **education, skills, experience & projects** — powered by Ayesha's CV")

if os.path.exists(CV_PATH):
    with open(CV_PATH, "rb") as f:
        st.download_button(
            label="📄 Download Full CV",
            data=f,
            file_name="Ayesha_Zafar_CV.pdf",
            mime="application/pdf"
        )

# Avatars
BOT_AVATAR = "https://cdn-icons-png.flaticon.com/512/4712/4712107.png"
USER_AVATAR = "https://cdn-icons-png.flaticon.com/512/1077/1077063.png"

# ── Suggestion Buttons (only at start) ────────────────────────
if len(st.session_state.messages) == 0:
    st.markdown("#### Quick questions to try:")
    cols = st.columns(3)
    suggestions = [
        "Tell me about Ayesha's projects",
        "What internships has Ayesha done?",
        "What are Ayesha's strongest technical skills?"
    ]
    
    for col, text in zip(cols, suggestions):
        if col.button(text):
            st.session_state.messages.append({"role": "user", "content": text})
            with st.spinner("Thinking..."):
                result = qa_chain.invoke({"question": text})
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"]
            })
            st.rerun()

# ── Chat Input ────────────────────────────────────────────────
if prompt := st.chat_input("Ask anything about Ayesha's CV..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("Thinking..."):
        result = qa_chain.invoke({"question": prompt})
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"]
    })

# ── Chat Messages ─────────────────────────────────────────────
st.markdown('<div class="chat-wrapper" id="chatbox">', unsafe_allow_html=True)

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

# ── Reliable Auto-Scroll (best practice 2025) ─────────────────
st.markdown("""
<script>
    function scrollToBottom() {
        const chatbox = document.getElementById("chatbox");
        if (chatbox) {
            chatbox.scrollTop = chatbox.scrollHeight;
        }
    }
    
    // Run immediately
    scrollToBottom();
    
    // Run again after small delay (content loading)
    setTimeout(scrollToBottom, 150);
    
    // Watch for any new content
    new MutationObserver(scrollToBottom)
        .observe(document.getElementById("chatbox") || document.body, {
            childList: true,
            subtree: true
        });
</script>
""", unsafe_allow_html=True)
