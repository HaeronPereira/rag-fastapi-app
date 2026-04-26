from google import genai
from app.config import API_KEY, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
import os
import pickle
import numpy as np
from pypdf import PdfReader
from app.logger import get_logger
logger = get_logger(__name__)



def load_data(file_path: str) -> str:
    try:
        if file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                logger.info(f"Loaded text length: {len(text)}")
                return text

        elif file_path.endswith(".pdf"):
            reader = PdfReader(file_path)
            text = ""

            for page in reader.pages:
                text += page.extract_text() + "\n"
            logger.info(f"Loaded text length: {len(text)}")
            return text

        else:
            logger.error("Unsupported file type: %s", file_path)
            raise ValueError("Unsupported file type")
    except Exception:
        logger.exception("Failed to load data")
        raise


def create_embeddings(client, texts):
    emb = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts
    )
    vectors=[e.values for e in emb.embeddings]
    return np.array(vectors).astype("float32")


def build_vector_store(text, file_path):
    file_id = os.path.splitext(os.path.basename(file_path))[0]
    INDEX_FILE = f"data/faiss_{file_id}.index"
    DOCS_FILE = f"data/docs_{file_id}.pkl"

    if os.path.exists(INDEX_FILE) and os.path.exists(DOCS_FILE):
        logger.info("Loading cached index...")
        index = faiss.read_index(INDEX_FILE)

        with open(DOCS_FILE, "rb") as f:
            docs = pickle.load(f)

        return docs, index
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    docs = splitter.split_text(text)
    logger.info(f"Created {len(docs)} chunks")

    client = genai.Client(api_key=API_KEY)

    embeddings = create_embeddings(client, docs)

    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)

    with open(DOCS_FILE, "wb") as f:
        pickle.dump(docs, f)

    logger.info("Saved index to disk")

    return docs, index