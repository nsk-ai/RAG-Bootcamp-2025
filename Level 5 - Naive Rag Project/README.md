# GROQ RAG ChatBot – Chat with Your PDFs

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot that lets you **upload a PDF** and **interact with their content** using **GROQ LLM** and an **in-memory Chroma vector store**.

---

## 🚀 Features

- 📄 Upload and process **a PDF file**
- 🧠 Store document embeddings in **Chroma** (in-memory)
- 💬 Query with **GROQ LLM** using **RAG**
- 🔍 Inspect vector store chunks from the sidebar
- 🛠️ Modular, well-commented code for easy customization

---

## 🛠 Setup Instructions

### 1. Create a Virtual Environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up GROQ API Key

To use this RAG chatbot, you'll need a GROQ API key.

#### Get Your API Key

1. Visit [console.groq.com/keys](https://console.groq.com/keys)
2. Sign up or log in to your GROQ account
3. Create a new API key
4. Copy the generated API key

#### Set Environment Variable

**Windows (Command Prompt)**

```cmd
set GROQ_API_KEY=your_api_key_here
```

**Windows (PowerShell)**

```powershell
$env:GROQ_API_KEY="your_api_key_here"
```

**macOS / Linux**

```bash
export GROQ_API_KEY="your_api_key_here"
```

**Alternative: Create a .env file**

You can also create a `.env` file in the project root directory:

```
GROQ_API_KEY=your_api_key_here
```

> ⚠️ **Important**: Never commit your API key to version control. Add `.env` to your `.gitignore` file if using this method.

---

## ▶️ Run the App

After installing dependencies, start the Streamlit app:

```bash
streamlit run index.py
```

The app will open in your default web browser at:

```
http://localhost:8501
```

If it doesn’t open automatically, copy and paste the URL from your terminal into your browser.

> 💡 Build your own custom RAG chatbot effortlessly!
