import streamlit as st
import os
from dotenv import load_dotenv

# ---- LangChain Community Packages ----
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ---- Memory + Chains ----
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# ---- Groq LLM ----
from langchain_groq import ChatGroq

# ---------------- Page Setup ----------------
st.set_page_config(
    page_title="Ayesha's Career Chatbot",
    layout="centered"
)

# ---------------- Load Environment ----------------
load_dotenv()
if "GROQ_API_KEY" not in os.environ:
    st.error("❌ GROQ_API_KEY not found. Add it in your .env file.")
    st.stop()

CV_PATH = "cv.pdf"
INDEX_DIR = "chroma_index"

# ---------------- Vector Store ----------------
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

# ---------------- Load AI ----------------
vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 12})

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

# ---------------- Session State ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "show_suggestions" not in st.session_state:
    st.session_state.show_suggestions = True

# ---------------- Custom CSS (CHAT ONLY) ----------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to bottom, #0f0f0f, #1a1a1a);
        min-height: 100vh;
    }

    .chat-row {
        display: flex;
        gap: 10px;
        margin-bottom: 12px;
        align-items: flex-start;
    }

    .chat-avatar {
        width: 38px;
        height: 38px;
        border-radius: 50%;
        flex-shrink: 0;
    }

    .user-bubble, .bot-bubble {
        padding: 12px 14px;
        border-radius: 16px;
        max-width: 78%;
        font-size: 15px;
        line-height: 1.5;
        word-wrap: break-word;
    }

    .user-bubble {
        background: #2563eb;
        color: white;
    }

    .bot-bubble {
        background: rgba(255,255,255,0.08);
        color: white;
    }

    /* Mobile optimization (chat only) */
    @media (max-width: 600px) {
        .user-bubble, .bot-bubble {
            max-width: 92%;
            font-size: 14px;
        }

        .chat-avatar {
            width: 32px;
            height: 32px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- UI Header ----------------
st.title("✨ Ask Ayesha's AI Career Bot")
st.write(
    "Ask anything about **Ayesha's education, skills, projects, and experience**. "
    "Responses are generated **only from her CV**."
)

BOT_AVATAR = "https://cdn-icons-png.flaticon.com/512/4712/4712107.png"
USER_AVATAR = "https://cdn-icons-png.flaticon.com/512/1077/1077063.png"

# ---------------- Suggestion Buttons (DEFAULT STYLE) ----------------
if st.session_state.show_suggestions:
    st.markdown("**Try asking:**")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🎓 What is Ayesha's education?"):
            st.session_state.messages.append(
                {"role": "user", "content": "What is Ayesha's education?"}
            )
            st.session_state.show_suggestions = False
            st.rerun()

    with col2:
        if st.button("💼 What projects has Ayesha done?"):
            st.session_state.messages.append(
                {"role": "user", "content": "What projects has Ayesha done?"}
            )
            st.session_state.show_suggestions = False
            st.rerun()

# ---------------- Chat Input ----------------
if prompt := st.chat_input("Type your question about Ayesha's CV..."):
    st.session_state.show_suggestions = False
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.spinner("Thinking..."):
        result = qa_chain.invoke({"question": prompt})
        answer = result["answer"]

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

# ---------------- Display Chat ----------------
chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "assistant":
            st.markdown(
                f"""
                <div class="chat-row">
                    <img src="{BOT_AVATAR}" class="chat-avatar">
                    <div class="bot-bubble">{msg['content']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="chat-row">
                    <img src="{USER_AVATAR}" class="chat-avatar">
                    <div class="user-bubble">{msg['content']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
