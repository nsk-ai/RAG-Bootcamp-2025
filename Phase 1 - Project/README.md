# YouTube Video Q\&A with RAG 💬

This is an interactive Q\&A application that allows you to "chat" with any YouTube video. It uses a Retrieval-Augmented Generation (RAG) pipeline to understand the video's content and answer your questions based on the transcript.

[GIF of the app in action]

## Features

- **Interactive Q\&A:** Ask questions about a YouTube video in natural language.
- **Advanced Retrieval:** Uses a sophisticated retrieval pipeline with a reranker (**Cross-Encoder**) for highly accurate context finding.
- **Fast Generation:** Powered by the incredibly fast **Groq** API with Llama 3 for near-instant answers.
- **Open-Source Embeddings:** Utilizes a local, open-source model from Hugging Face for text embeddings.
- **Simple UI:** Built with **Streamlit** for a clean and easy-to-use web interface.

## Tech Stack

- **Framework:** LangChain
- **UI:** Streamlit
- **LLM:** Groq (Llama 3 8B)
- **Embedding Model:** Hugging Face `all-MiniLM-L6-v2`
- **Vector Store:** ChromaDB (in-memory)
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Data Loader:** `yt-dlp`

---

## Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

- Python 3.8 or higher
- Git

### 1\. Clone the Repository

First, clone the project repository to your local machine.

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2\. Create a Python Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

```bash
# Create the virtual environment
python3 -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3\. Install Dependencies

This project's dependencies are listed in `requirements.txt`.

**(First, if you haven't created a `requirements.txt` file yet, run this command):**

```bash
pip freeze > requirements.txt
```

Now, install the required packages using pip:

```bash
pip install -r requirements.txt
```

### 4\. Set Up Environment Variables

The application requires an API key from Groq to use its LLM.

1.  Create a file named `.env` in the root of your project directory. You can do this by copying the example file:

    ```bash
    cp .env.example .env
    ```

    _(If you don't have a `.env.example` file, simply create a new file named `.env`)_

2.  Get your API key from the [GroqCloud Console](https://console.groq.com/keys).

3.  Open the `.env` file and add your API key like this:

    ```
    GROQ_API_KEY="gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    ```

---

## Usage

With the environment set up and dependencies installed, you can now run the Streamlit application.

```bash
streamlit run app.py
```

Your web browser should automatically open with the application running.

1.  Paste a YouTube video URL into the sidebar input field.
2.  Click the **"Process Video"** button and wait for the processing to complete.
3.  Once processed, you can ask questions about the video in the main input field.

---

## Project Structure

The project is organized in a modular way to keep the code clean and maintainable.

```
.
├── helpers/
│   ├── __init__.py       # Makes 'helpers' a Python package
│   ├── chain.py          # Creates the final RAG chain with the LLM
│   ├── chunker.py        # Splits documents into smaller chunks
│   ├── retriever.py      # Creates the retriever and reranker
│   ├── vectorstore.py    # Creates the ChromaDB vector store
│   └── youtubeloader.py  # Loads and cleans transcripts using yt-dlp
├── .env                  # Stores API keys (secret, not committed to git)
├── .gitignore            # Specifies files for git to ignore
├── app.py                # The main Streamlit application file
├── requirements.txt      # Project dependencies
└── README.md             # This file
```

---

## How It Works

The application follows a standard RAG pipeline:

1.  **Ingestion:** The `youtubeloader` fetches the video transcript using `yt-dlp` and cleans it.
2.  **Chunking:** The `chunker` splits the clean transcript into smaller, overlapping documents.
3.  **Indexing:** The `vectorstore` helper uses the `all-MiniLM-L6-v2` model to create a numerical vector (embedding) for each chunk and stores them in a Chroma vector database.
4.  **Retrieval & Reranking:** When a question is asked, the `retriever` finds the most relevant chunks from ChromaDB. These results are then re-scored by the Cross-Encoder for higher accuracy.
5.  **Generation:** The top-ranked chunks and the original question are passed to the Groq LLM within a structured prompt, which then generates the final, grounded answer.
