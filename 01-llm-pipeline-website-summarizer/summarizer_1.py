import sys
from llm_client import get_client
from scraper import fetch_website_contents

client, MODEL = get_client()

SYSTEM_PROMPT = """You analyze the contents of a website and give a short,
friendly summary. Ignore navigation menus and boilerplate.
Respond in markdown."""


def summarize(url: str) -> str:
    website = fetch_website_contents(url)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Summarize this website:\n\n{website}"},
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else input("Enter URL: ")
    print(summarize(url))