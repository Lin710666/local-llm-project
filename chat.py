
import json
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


API_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:4b"

# Instructions that define the assistant's behavior.
SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are a helpful AI assistant and English teacher. "
        "Explain concepts clearly and simply. "
        "Use examples when helpful. "
        "Gently correct the user's English when appropriate."
    ),
}


def get_reply(messages):
    """Send the conversation to Ollama and stream the reply."""

    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": True,
        "think": False,
        "keep_alive": "10m",
        "options": {
            "num_predict": 256
        },
    }

    request = Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    answer_parts = []

    with urlopen(request, timeout=300) as response:
        for line in response:
            if not line.strip():
                continue

            chunk = json.loads(line.decode("utf-8"))
            message = chunk.get("message", {})
            content = message.get("content", "")

            if content:
                print(content, end="", flush=True)
                answer_parts.append(content)

            if chunk.get("done"):
                break

    print()
    return "".join(answer_parts)


def main():
    # This list stores the conversation for the current session.
    messages = [SYSTEM_MESSAGE.copy()]

    print("Local AI Chatbot")
    print("Commands: /clear to reset, /exit to quit")

    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_text:
            continue

        if user_text.lower() == "/exit":
            print("Goodbye!")
            break

        if user_text.lower() == "/clear":
            messages = [SYSTEM_MESSAGE.copy()]
            print("Conversation history cleared.")
            continue

        # 1. Save the user's message.
        messages.append({
            "role": "user",
            "content": user_text,
        })

        try:
            print("AI: ", end="", flush=True)

            # 2. Send all messages and receive the response.
            answer = get_reply(messages)

            # 3. Save the assistant's response.
            messages.append({
                "role": "assistant",
                "content": answer,
            })

        except HTTPError as error:
            print(f"\nHTTP error: {error.code} {error.reason}")
            print(error.read().decode("utf-8", errors="replace"))
            messages.pop()

        except (URLError, TimeoutError) as error:
            print(f"\nConnection or timeout error: {error}")
            messages.pop()

        except (json.JSONDecodeError, KeyError) as error:
            print(f"\nUnexpected response from Ollama: {error}")
            messages.pop()


if __name__ == "__main__":
    main()
