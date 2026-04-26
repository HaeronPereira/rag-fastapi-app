from google import genai
import os
from dotenv import load_dotenv

def main():
    load_dotenv()  # 👈 load .env file

    api_key = os.getenv("GOOGLE_CLOUD_API_KEY")

    if not api_key:
        print("❌ API key not found")
        return

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents="Say hello in one short sentence"
    )

    print("✅ Response:")
    print(response.text)


if __name__ == "__main__":
    main()