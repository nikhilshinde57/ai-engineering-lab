import json
from llm_client import get_client

client, MODEL = get_client()

# ---------- Part 1: the tools are plain Python functions ----------
PRICES = {"shoes": 799, "hat": 399, "bag": 1420, "shorts": 1299, "pants": 1699}
STOCK = {"shoes": 12, "hat": 0, "bag": 5, "shorts": 3, "pants": 0}


def get_price(item: str) -> str:
    print(f"🔧 tool called: get_price({item})")
    price = PRICES.get(item.lower())
    return f"₹{price}" if price is not None else "unknown item"


def check_stock(item: str) -> str:
    print(f"🔧 tool called: check_stock({item})")
    qty = STOCK.get(item.lower())
    if qty is None:
        return "unknown item"
    return f"{qty} in stock" if qty > 0 else "out of stock"


# Registry: tool name -> function. Adding a tool = one line here + one menu card.
TOOL_REGISTRY = {"get_price": get_price, "check_stock": check_stock}


# ---------- Part 2: the "menu cards" the model reads ----------
def tool_spec(name: str, description: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,        # the model decides from THIS sentence
            "parameters": {
                "type": "object",
                "properties": {"item": {"type": "string", "description": "the item name"}},
                "required": ["item"],
            },
        },
    }


TOOLS = [
    tool_spec("get_price", "Get the price of a shop item the user asks about."),
    tool_spec("check_stock", "Check whether a shop item is in stock and how many are left."),
]

SYSTEM_PROMPT = (
    "You are a friendly shop assistant. Use the tools for any price or stock "
    "question; never guess prices or stock. Answer small talk directly."
)

MAX_STEPS = 5   # guard so a confused model can't loop forever


# ---------- Part 3: the loop: think -> maybe call tools -> answer ----------
def agent(user_message: str, history: list | None = None) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history or []                                # memory: re-send past turns
    messages.append({"role": "user", "content": user_message})

    for _ in range(MAX_STEPS):
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS
        )
        msg = response.choices[0].message

        if not msg.tool_calls:                               # no tool needed -> done
            return msg.content

        messages.append(msg.model_dump(exclude_none=True))   # keep the model's request
        for call in msg.tool_calls:
            fn = TOOL_REGISTRY.get(call.function.name)
            args = json.loads(call.function.arguments)
            result = fn(**args) if fn else f"unknown tool {call.function.name}"
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": result}
            )

    return "Sorry, I couldn't finish that request. Please try rephrasing."


if __name__ == "__main__":
    print(agent("How much are the shoes?"))
    print("---")
    print(agent("Hi! What can you help with?"))
    print("---")
    print(agent("Is the hat in stock, and what does the bag cost?"))