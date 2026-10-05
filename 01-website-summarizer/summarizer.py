import sys
from llm_client import get_client
from scraper import fetch_website_contents

client, MODEL = get_client()

MAX_CHARS = 20_000

PERSONALITIES = {
    
    "Friendly": "give a short, friendly summary",
    "Snarky": "give a snarky, humorous summary",
    "Explain like I'm 10": "explain it simply, as if to a 10-year-old",
    "Executive brief": (
        "respond with ONLY 3-5 crisp bullet points a busy executive would want "
        "and nothing else: no intro sentence, no closing paragraph, no headers"
    ),
}


def build_system_prompt(personality: str) -> str:
    style = PERSONALITIES.get(personality, PERSONALITIES["Friendly"])
    return (
        f"You analyze the contents of a website and {style}. "
        "Ignore navigation menus and boilerplate. Respond in markdown."
    )


def summarize(url: str, personality: str = "Friendly") -> str:
    website = fetch_website_contents(url)[:MAX_CHARS]
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": build_system_prompt(personality)},
            {"role": "user", "content": f"Summarize this website:\n\n{website}"},
        ],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else input("Enter URL: ")
    print(summarize(url))