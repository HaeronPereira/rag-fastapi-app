from google import genai
from app.config import API_KEY, LLM_MODEL
from app.logger import get_logger
logger = get_logger(__name__)

def generate_answer(context, query):
    client = genai.Client(api_key=API_KEY)

    prompt = f"""
    You are a helpful assistant.

    Answer ONLY using the provided context.
    If answer is not in context, say "I don't know".    

    Context:
    {context}

    Question:
    {query}
    """
    try:
        response = client.interactions.create(
            model=LLM_MODEL,
            input=prompt
        )
        logger.info("Sending prompt to LLM...")
        for out in response.outputs:
            if hasattr(out, "text") and out.text:
                return out.text
        return "No valid text response found"
    except Exception:
        logger.exception("LLM Generation failed")
        return "LLM Error occured"