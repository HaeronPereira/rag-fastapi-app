from app.pipeline import RAGPipeline
from app.logger import get_logger
logger = get_logger(__name__)
def main():
    logger.info("RAG App Started")  
    try:
        rag = RAGPipeline("data/rag_details.pdf")
    except Exception:
        logger.exception("Failed to initialize pipeline")

    while True:
        try:
            q = input("\nAsk: ")

            if q.lower() == "exit":
                print("Exiting...")
                break

            ans = rag.query(q)
            print("\nAnswer:", ans)
        except Exception:
            logger.exception("Error during query")

if __name__ == "__main__":
    main()