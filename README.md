# 🤖 Ayesha’s Career Chatbot — RAG Powered AI Assistant

Ayesha’s Career Chatbot is an AI-powered assistant built using a **Retrieval-Augmented Generation (RAG)** pipeline.
It answers questions strictly based on information retrieved from Ayesha Zafar’s CV, ensuring **accurate, grounded, and hallucination-free responses**.

The chatbot allows users (recruiters, mentors, peers) to explore her **education, skills, experience, and projects** through natural language conversations.

🌐 **Live App:**
[https://ayeshay-career-chatbot.streamlit.app/](https://ayeshay-career-chatbot.streamlit.app/)

📂 **GitHub Repo:**
[https://github.com/Ayesha-Zafar-03/ayeshay-career-chatbot](https://github.com/Ayesha-Zafar-03/ayeshay-career-chatbot)

---

## 🚀 Features

* 📄 CV-based question answering (no generic answers)
* 🧠 Retrieval-Augmented Generation (RAG) pipeline
* 🔍 Semantic search using vector embeddings
* 💬 Conversational memory
* ⚡ Fast inference using Groq LLaMA-3
* 🎨 Interactive Streamlit UI
* ☁️ Deployed on Streamlit Cloud

---

## 🏗️ System Architecture

```
PDF CV → Text Chunking → Embeddings → ChromaDB (Vector Store)
                                      ↓
User Question → Similarity Search → Retrieved Chunks → LLM → Answer
```

The model is instructed to respond **only from retrieved CV content**.

---

## 🛠️ Tech Stack

**Frontend & App Framework**

* Streamlit

**LLM & Orchestration**

* Groq (LLaMA-3)
* LangChain

**RAG Components**

* PyPDFLoader
* RecursiveCharacterTextSplitter
* HuggingFace Embeddings
* ChromaDB (Vector Database)

**Programming Language**

* Python

**Deployment**

* Streamlit Cloud

---

## 📂 Project Structure

```
├── app.py
├── requirements.txt
├── Ayesha_Zafar_CV.pdf
├── chroma_db/              # vector database (auto-generated)
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Ayesha-Zafar-03/ayeshay-career-chatbot.git
cd ayeshay-career-chatbot
```

---

### 2️⃣ Create virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Add environment variables

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

---

### 5️⃣ Run the app

```bash
streamlit run app.py
```

---

## 🔄 Updating the CV (Important)

If you update the CV PDF, you **must rebuild the vector database**.

Delete the old Chroma folder before rerunning:

```bash
rm -rf chroma_db
```

(or manually delete it)

Then rerun the app so new embeddings are created.

---

## 🧪 Example Questions

* “What projects has Ayesha worked on?”
* “What machine learning skills does she have?”
* “Tell me about her internship experience.”
* “What technologies does she use for AI apps?”
* “Has she worked on RAG-based systems?”

---

## 🎯 Use Cases

* Personal AI portfolio assistant
* Recruiter-friendly interactive CV
* Demonstration of RAG pipelines
* LLM + vector database integration example

---

## 👩‍💻 Author

**Ayesha Zafar**
BSc Software Engineering | AI & Machine Learning Enthusiast

* GitHub: [https://github.com/Ayesha-Zafar-03](https://github.com/Ayesha-Zafar-03)
* LinkedIn: [https://www.linkedin.com/in/ayesha-zafar03](https://www.linkedin.com/in/ayesha-zafar03)
* Kaggle: [https://www.kaggle.com/ayeshayzafar](https://www.kaggle.com/ayeshayzafar)

---

## 📜 License

This project is open-source and available for educational and portfolio use.

---
