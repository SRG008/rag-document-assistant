
import time

from google.genai.errors import ServerError

from app.core.config import GEMINI_MODEL
from app.core.gemini_client import client


def ask_gemini(question: str) -> str:
    """
    Send a question to Gemini.

    Retry temporary server errors with
    exponential backoff.
    """

    max_attempts = 3

    for attempt in range(1, max_attempts + 1):

        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=question
            )

            return response.text or ""

        except ServerError as error:

            if error.code != 503 or attempt == max_attempts:
                raise

            wait_seconds = 2 ** attempt

            print(
                f"Gemini temporarily unavailable. "
                f"Retrying in {wait_seconds} seconds..."
            )

            time.sleep(wait_seconds)

    return ""


def main():

    question = "Explain what Retrieval Augmented Generation is."

    answer = ask_gemini(question)

    print("\nQUESTION:")
    print(question)

    print("\nGEMINI RESPONSE:")
    print(answer)


if __name__ == "__main__":
    main()
