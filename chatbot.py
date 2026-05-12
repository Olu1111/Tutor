import os

from conversation import (
    DEFAULT_MODEL,
    build_client,
    new_conversation,
    submit_user_message,
)


def resolve_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key:
        return api_key

    raise RuntimeError(
        "Missing OpenAI API key. Set OPENAI_API_KEY in your environment before running chatbot.py."
    )


def main() -> None:
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    api_key = resolve_api_key()
    client = build_client(api_key)
    state = new_conversation()

    print("Tutor chatbot ready. Ask a Python question.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            print("\nGoodbye!")
            break

        if not user_input:
            print("Please enter a Python question.\n")
            continue

        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        try:
            answer = submit_user_message(client, state, user_input, model)
        except Exception as error:
            print(f"Tutor: Sorry, I hit an API error: {error}\n")
            continue
        print(f"Tutor: {answer}\n")
        print(f"Progress: {state.progress.to_summary()}\n")


if __name__ == "__main__":
    main()
