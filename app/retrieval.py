import numpy as np
from app.logger import get_logger
logger = get_logger(__name__)


def retrieve(query, client, index, docs, embedding_model, top_k=3):
    try:
        emb = client.models.embed_content(
            model=embedding_model,
            contents=query   
        )

        q_emb = np.array([emb.embeddings[0].values]).astype("float32") 
        logger.info(f"Retrieving top {top_k} chunks for query: {query}")

        distances, indices = index.search(q_emb, top_k)

        return [docs[i] for i in indices[0] if i >= 0]
    except Exception:
        logger.exception("Retrieval Failed")
        return []