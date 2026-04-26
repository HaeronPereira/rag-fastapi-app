import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOOGLE_CLOUD_API_KEY")

EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash-lite"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
