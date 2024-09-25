# Load .env file using:
from dotenv import load_dotenv
load_dotenv()
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = "llama3-groq-8b-8192-tool-use-preview"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
