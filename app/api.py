from fastapi import FastAPI
from pydantic import BaseModel
from app.pipeline import RAGPipeline

app = FastAPI()

# Initialize once
rag = RAGPipeline("data/rag_details.pdf")


class QueryRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {"status": "running"}


@app.post("/query")
def query_rag(req: QueryRequest):
    answer = rag.query(req.question)
    return {"answer": answer}