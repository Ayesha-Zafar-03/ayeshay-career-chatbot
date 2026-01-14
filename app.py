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

    loader = PyPDFLoader(CV_PATH) if CV_PATH.lower().endswith(".pdf") else TextLoader(CV_PATH, encoding="utf8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    return Chroma.from_documents(chunks, embeddings, persist_directory=INDEX_DIR)

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
st.markdown("""
<style>

.stApp {
    background: url("https://c.tenor.com/Ho0ZextTZJEAAAAC/ai-digital.gif")
    no-repeat center center fixed;
    background-size: cover;
}

h1 { color: white !important; }

.chat-wrapper {
    height: 70vh;
    overflow-y: auto;
    padding: 15px;
    background: rgba(0,0,0,0.45);
    border-radius: 18px;
}

/* Chat rows */
.chat-row {
    display: flex;
    margin-bottom: 12px;
    align-items: flex-end;
}

/* Bot left */
.chat-row.bot {
    justify-content: flex-start;
}

/* User right */
.chat-row.user {
    justify-content: flex-end;
}

/* Bubbles */
.user-bubble, .bot-bubble {
    border-radius: 16px;
    padding: 10px 14px;
    max-width: 65%;
    animation: fadeIn 0.3s ease-in-out;
    word-wrap: break-word;
}

/* User */
.user-bubble {
    background: #1E1E1E;
    color: white;
    border-bottom-right-radius: 4px;
}

/* Bot */
.bot-bubble {
    background: rgba(0,0,0,0.75);
    color: white;
    border-bottom-left-radius: 4px;
}

/* Avatars */
.chat-avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    margin: 0 8px;
    border: 2px solid black;
}

/* Input */
.stChatInput input {
    background-color: #1E1E1E !important;
    color: white !important;
    border: 2px solid black !important;
    border-radius: 10px !important;
}

/* Buttons */
.stButton>button {
    background-color: #1E1E1E !important;
    color: white !important;
    border-radius: 8px !important;
    border: 1px solid black !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------- UI ----------------
st.title("✨ Ask Ayesha's AI Career Bot")
st.write("Ask me anything about **Ayesha's education, skills, and projects** — answers come only from her CV.")

# ---------------- Download CV ----------------
if os.path.exists(CV_PATH):
    with open(CV_PATH, "rb") as file:
        st.download_button("📄 Download CV", file, "Ayesha_Zafar_CV.pdf", "application/pdf")

BOT_AVATAR = "https://cdn-icons-png.flaticon.com/512/4712/4712107.png"
USER_AVATAR = "https://cdn-icons-png.flaticon.com/512/1077/1077063.png"

# ---------------- Suggestions ----------------
if len(st.session_state.messages) == 0:
    st.markdown("#### 🔎 Try asking me:")
    cols = st.columns(3)
    suggestions = [
        "Tell me about Ayesha's projects",
        "What internships does Ayesha have?",
        "What are Ayesha's top technical skills?"
    ]
    for i, text in enumerate(suggestions):
        if cols[i].button(text):
            st.session_state.messages.append({"role": "user", "content": text})
            with st.spinner("Thinking..."):
                result = qa_chain.invoke({"question": text})
            st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

# ---------------- Chat Input ----------------
if prompt := st.chat_input("Type your question about Ayesha's CV..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Thinking..."):
        result = qa_chain.invoke({"question": prompt})
    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

# ---------------- Chat Display ----------------
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

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Auto Scroll Chat Box ----------------
st.markdown("""
<script>
const chatbox = document.getElementById("chatbox");
if (chatbox) {
    chatbox.scrollTop = chatbox.scrollHeight;
}
</script>
""", unsafe_allow_html=True)
