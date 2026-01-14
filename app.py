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

# ---------------- Setup ----------------
st.set_page_config(
    page_title="Ayesha's Career Chatbot",
    layout="centered"
)

# ---------------- Environment ----------------
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

# ---------------- Session State ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- Custom CSS ----------------
st.markdown(
    """
    <style>
    .stApp {
        background: url("https://c.tenor.com/Ho0ZextTZJEAAAAC/ai-digital.gif")
        no-repeat center center fixed;
        background-size: cover;
        min-height: 100vh;
    }

    /* Title color */
    h1 {
        color: white !important;
    }

    .user-bubble, .bot-bubble {
        border-radius: 14px;
        padding: 10px 14px;
        margin: 6px 0;
        max-width: 75%;
        animation: fadeIn 0.3s ease-in-out;
    }

    .user-bubble {
        background: #1E1E1E;
        color: white;
    }

    .bot-bubble {
        background: rgba(0,0,0,0.6);
        color: white;
    }

    .chat-row {
        display: flex;
        margin-bottom: 10px;
    }

    .chat-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin-right: 12px;
        border: 2px solid black;
    }

    /* Chat input black background */
    .stChatInput input {
        background-color: #1E1E1E !important;
        color: white !important;
        border: 2px solid black !important;
        border-radius: 8px !important;
    }
    .stChatInput input:focus {
        border: 2px solid black !important;
        box-shadow: none !important;
    }

    /* Suggestion buttons black */
    .stButton>button {
        background-color: #1E1E1E !important;
        color: white !important;
        border-radius: 8px !important;
        border: 1px solid black !important;
        padding: 8px 16px !important;
    }
    .stButton>button:hover {
        border: 1px solid white !important;
        cursor: pointer;
    }

   
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- UI ----------------
st.title("✨ Ask Ayesha's AI Career Bot")
st.write(
    "Ask me anything about **Ayesha's education, skills, and projects** "
    "— answers come only from her CV."
)



BOT_AVATAR = "https://cdn-icons-png.flaticon.com/512/4712/4712107.png"
USER_AVATAR = "https://cdn-icons-png.flaticon.com/512/1077/1077063.png"

# ---------------- Suggestion Panel ----------------
if len(st.session_state.messages) == 0:
    st.markdown("#### 🔎 Try asking me:")

    cols = st.columns(3)
    suggestions = [
        "Tell me about Ayesha's projects",
        "What internships does Ayesha have?",
        "What are Ayesha's top technical skills?"
    ]

    for i, text in enumerate(suggestions):
        if cols[i].button(text, key=f"sugg_{i}"):
            st.session_state.messages.append(
                {"role": "user", "content": text}
            )

            with st.spinner("Thinking..."):
                result = qa_chain.invoke({"question": text})
                answer = result["answer"]

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
            st.rerun()

# ---------------- Chat Input ----------------
if prompt := st.chat_input("Type your question about Ayesha's CV..."):
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
