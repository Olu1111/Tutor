# Tutor

Python-question chatbot with both a command-line interface and a Streamlit web UI.
It now keeps short-term conversation memory and tracks basic student progress.
It also includes a safe, built-in auto-evaluator for practice code.

## Setup

1. Put your API key in `tutor_key.txt`.
2. Install dependencies:

	```bash
	pip install -r requirements.txt
	```

3. Run the command-line chatbot:

	```bash
	python chatbot.py
	```

Type `quit` or `exit` to stop the chat.

## Web UI

Run the Streamlit interface:

```bash
streamlit run streamlit_app.py
```

This opens a chat UI where users can ask Python questions, see message history, and clear the conversation.

The sidebar also shows conversation memory, topic counts, confirmations, and struggle signals so the tutor can adapt to the learner.

## Auto-evaluator

Open the `Code evaluator` tab to paste student code and run it against a small set of guarded practice tests.
The evaluator uses AST checks, a restricted built-in environment, and a short timeout to reduce risk.
It supports common numeric and sequence helpers such as `sorted`, `min`, `max`, `sum`, `pow`, and `divmod`, plus a small whitelist of safe collection methods like list `append`, `pop`, `sort`, and dict `get`.

Note: this is a conservative safety layer, not a perfect sandbox. Avoid using it for untrusted production code.
