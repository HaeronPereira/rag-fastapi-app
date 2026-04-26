from google import genai
from app.ingestion import load_data, build_vector_store
from app.retrieval import retrieve
from app.generation import generate_answer
from app.config import API_KEY, EMBEDDING_MODEL


class RAGPipeline:
    def __init__(self, file_path):
        text = load_data(file_path)
        self.docs, self.index = build_vector_store(text,file_path)
        self.client = genai.Client(api_key=API_KEY)

    def query(self, question):
        chunks = retrieve(
            question,
            self.client,
            self.index,
            self.docs,
            EMBEDDING_MODEL
        )

        context = " ".join(chunks)

        return generate_answer(context, question)