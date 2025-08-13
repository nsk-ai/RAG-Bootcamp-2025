# config.py

import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = "pdf-chat-index"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
