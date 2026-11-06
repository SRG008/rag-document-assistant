import os

from dotenv import load_dotenv


load_dotenv()


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)

GEMINI_EMBEDDING_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL",
    "gemini-embedding-001"
)


if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing. "
        "Please configure it in the .env file."
    )