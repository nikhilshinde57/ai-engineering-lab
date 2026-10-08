from urllib.parse import urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup

# Pretend to be a normal browser; many sites block the default Python user agent
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def fetch_website_contents(url: str) -> str:
    """Fetch a URL and return its readable text (title + body)."""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    # Drop query params and fragments (tracking params like gclid add noise)
    scheme, netloc, path, _query, _fragment = urlsplit(url)
    url = urlunsplit((scheme, netloc, path, "", ""))

    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return f"Could not fetch the website. Error: {e}"

    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.title.string if soup.title else "No title found"

    # Strip elements that are noise for summarization
    for tag in soup(["script", "style", "nav", "footer", "header", "img", "input"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    return f"Title: {title}\n\nPage contents:\n{text}"


if __name__ == "__main__":
    # Quick manual test
    print(fetch_website_contents("https://en.wikipedia.org/wiki/Large_language_model")[:1000])