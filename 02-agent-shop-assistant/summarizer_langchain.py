from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm_client import get_langchain_model
from scraper import fetch_website_contents

prompt = ChatPromptTemplate.from_template(              # 1. prompt with a {blank}
    "Give a short, friendly summary of this website:\n\n{website}"
)
model = get_langchain_model(temperature=0.3)            # 2. the model
parser = StrOutputParser()                              # 3. AIMessage -> plain text

chain = prompt | model | parser                         # 4. read "|" as "then"


def summarize(url: str) -> str:
    website = fetch_website_contents(url)[:20_000]
    return chain.invoke({"website": website})           # 5. fill the blank and run


if __name__ == "__main__":
    print(summarize("https://en.wikipedia.org/wiki/Large_language_model"))