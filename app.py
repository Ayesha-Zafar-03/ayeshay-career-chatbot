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
st.set_page_config(page_title="Ayesha's Career Chatbot", layout="centered")

load_dotenv()
if "GROQ_API_KEY" not in os.environ:
    st.error("❌ GROQ_API_KEY not found.")
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

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

# ---------------- LLM ----------------
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

# ---------------- CSS ----------------
st.markdown(
    """
<style>
.stApp {
    background: url("https://c.tenor.com/Ho0ZextTZJEAAAAC/ai-digital.gif")
    no-repeat center center fixed;
    background-size: cover;
    min-height: 100vh;
    color: #EAF2FF;
    position: relative;
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.45);
    z-index: 0;
}

.stApp > * {
    position: relative;
    z-index: 1;
}

.chat-row {
    display: flex;
    align-items: flex-start;
    margin-bottom: 10px;
}

.chat-row.bot {
    justify-content: flex-start;
}

.chat-row.user {
    justify-content: flex-end;
}

.chat-avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    border: 2px solid black;
    flex-shrink: 0;
}

.chat-row.user .chat-avatar {
    order: 2;
    margin-left: 10px;
}

.chat-row.bot .chat-avatar {
    margin-right: 10px;
}

.user-bubble,
.bot-bubble {
    border-radius: 16px;
    padding: 10px 14px;
    max-width: 70%;
    word-wrap: break-word;
    animation: fadeIn 0.3s ease-in-out;
}

.user-bubble {
    background: #1E1E1E;
    color: white;
    border-bottom-right-radius: 4px;
}

.bot-bubble {
    background: rgba(0,0,0,0.65);
    color: white;
    border-bottom-left-radius: 4px;
}

.stChatInput input {
    background-color: black !important;
    color: white !important;
    border-radius: 12px !important;
    border: 1px solid black !important;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ---------- Mobile ---------- */
@media (max-width: 768px) {

    .user-bubble,
    .bot-bubble {
        max-width: 85%;
        font-size: 14px;
    }

    .chat-avatar {
        width: 32px;
        height: 32px;
    }

    .stChatInput {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 8px 12px;
        background: rgba(0,0,0,0.9);
        z-index: 100;
    }

    .stApp {
        padding-bottom: 90px;
    }

    h1 {
        font-size: 1.4rem !important;
        text-align: center;
    }
}
</style>
""",
    unsafe_allow_html=True
)

# ---------------- UI ----------------
st.title("✨ Ask Ayesha's AI Career Bot")
st.write(
    "Ask me anything about **Ayesha's education, skills, and projects** — "
    "answers come only from her CV."
)

BOT_AVATAR = "https://cdn-icons-png.flaticon.com/512/4712/4712107.png"
USER_AVATAR = "https://cdn-icons-png.flaticon.com/512/1077/1077063.png"

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
            row_class = "chat-row bot"
            avatar = BOT_AVATAR
            bubble = "bot-bubble"
        else:
            row_class = "chat-row user"
            avatar = USER_AVATAR
            bubble = "user-bubble"

        st.markdown(
            f"""
            <div class="{row_class}">
                <img src="{avatar}" class="chat-avatar">
                <div class="{bubble}">{msg['content']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
