from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List

from openai import OpenAI

SYSTEM_PROMPT = (
    "You are Tutor, a helpful Python teaching assistant. "
    "Answer Python questions clearly and accurately. "
    "Prefer short explanations first, then examples when useful. "
    "If the user shares buggy code, explain the bug and show a corrected version. "
    "Adapt to the learner's current progress and remember prior topics from this session."
)
DEFAULT_MODEL = "gpt-4.1-mini"
MAX_CONTEXT_MESSAGES = 10
SUMMARY_TRIGGER_MESSAGES = 8

TOPIC_KEYWORDS = {
    "syntax": ["syntax", "semicolon", "indent", "indentation", "parse", "error"],
    "functions": ["function", "def ", "parameter", "argument", "return"],
    "loops": ["for ", "while ", "loop", "iterate", "iteration"],
    "lists": ["list", "append", "pop", "slice", "index"],
    "dictionaries": ["dict", "dictionary", "key", "value", "mapping"],
    "classes": ["class", "object", "method", "inheritance", "oop"],
    "files": ["file", "open(", "read", "write", "csv", "json"],
    "data science": ["pandas", "numpy", "matplotlib", "dataframe", "plot"],
}
POSITIVE_PROGRESS_WORDS = {"got it", "understood", "understand", "solved", "worked", "thanks"}

MODEL_PRICING_PER_1M_TOKENS = {
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
}


@dataclass
class StudentProgress:
    questions_asked: int = 0
    topics_covered: Dict[str, int] = field(default_factory=dict)
    current_topic: str = ""
    confirmations: int = 0
    struggle_signals: int = 0

    def to_summary(self) -> str:
        if self.questions_asked == 0:
            return "No questions yet."

        topic_parts = [f"{topic} ({count})" for topic, count in self.topics_covered.items()]
        topic_text = ", ".join(topic_parts) if topic_parts else "No topics tracked yet."
        active_topic = self.current_topic or "not set"
        return (
            f"Questions asked: {self.questions_asked}. "
            f"Current topic: {active_topic}. "
            f"Topics covered: {topic_text}. "
            f"Success confirmations: {self.confirmations}. "
            f"Struggle signals: {self.struggle_signals}."
        )


@dataclass
class UsageEvent:
    timestamp_utc: str
    model: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float


@dataclass
class UsageStats:
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_estimated_cost_usd: float = 0.0
    call_count: int = 0
    recent_events: List[UsageEvent] = field(default_factory=list)

    def record(self, model: str, input_tokens: int, output_tokens: int, estimated_cost_usd: float) -> None:
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_estimated_cost_usd += estimated_cost_usd
        self.call_count += 1

        event = UsageEvent(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=estimated_cost_usd,
        )
        self.recent_events.append(event)
        self.recent_events = self.recent_events[-25:]


@dataclass
class ConversationState:
    summary: str = ""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    progress: StudentProgress = field(default_factory=StudentProgress)
    usage: UsageStats = field(default_factory=UsageStats)


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING_PER_1M_TOKENS.get(model)
    if pricing is None:
        return 0.0

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


def record_usage_from_response(usage: UsageStats, model: str, response: Any) -> None:
    response_usage = getattr(response, "usage", None)
    if response_usage is None:
        return

    input_tokens = int(getattr(response_usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(response_usage, "output_tokens", 0) or 0)
    estimated_cost = estimate_cost_usd(model, input_tokens, output_tokens)
    usage.record(model, input_tokens, output_tokens, estimated_cost)


def load_api_key_text(key_path) -> str:
    if not key_path.exists():
        raise FileNotFoundError("tutor_key.txt was not found in the project folder.")

    api_key = key_path.read_text(encoding="utf-8").strip()
    if not api_key:
        raise ValueError("tutor_key.txt is empty.")

    return api_key


def build_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def new_conversation() -> ConversationState:
    return ConversationState()


def build_system_prompt(summary: str, progress_summary: str) -> str:
    prompt = SYSTEM_PROMPT
    if summary:
        prompt += f"\n\nMemory from earlier in this session: {summary}"
    if progress_summary:
        prompt += f"\n\nStudent progress snapshot: {progress_summary}"
    return prompt


def build_context_messages(state: ConversationState) -> List[Dict[str, Any]]:
    context = [
        {"role": "system", "content": build_system_prompt(state.summary, state.progress.to_summary())}
    ]
    context.extend(state.messages[-MAX_CONTEXT_MESSAGES:])
    return context


def ask_chatbot(client: OpenAI, state: ConversationState, model: str) -> str:
    response = client.responses.create(model=model, input=build_context_messages(state))
    record_usage_from_response(state.usage, model, response)
    return response.output_text.strip()


def infer_topic(text: str) -> str:
    lowered = text.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return topic
    return "general Python"


def update_progress(progress: StudentProgress, user_input: str) -> None:
    progress.current_topic = infer_topic(user_input)
    progress.topics_covered[progress.current_topic] = progress.topics_covered.get(progress.current_topic, 0) + 1


def summarize_messages(
    client: OpenAI,
    model: str,
    old_messages: Iterable[Dict[str, Any]],
    existing_summary: str,
    usage: UsageStats,
) -> str:
    lines = []
    if existing_summary:
        lines.append(f"Previous memory: {existing_summary}")

    for message in old_messages:
        role = message.get("role", "unknown")
        content = str(message.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")

    prompt = (
        "Summarize this tutoring conversation memory for future context. "
        "Keep it concise, factual, and useful. Include the main topics, unresolved questions, "
        "and the learner's current needs. Return plain text only.\n\n"
        + "\n".join(lines)
    )

    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": prompt}],
    )
    record_usage_from_response(usage, model, response)
    return response.output_text.strip()


def trim_memory(client: OpenAI, state: ConversationState, model: str) -> None:
    if len(state.messages) <= SUMMARY_TRIGGER_MESSAGES:
        return

    preserve_count = 4
    old_messages = state.messages[:-preserve_count]
    if not old_messages:
        return

    try:
        state.summary = summarize_messages(client, model, old_messages, state.summary, state.usage)
    except Exception:
        fallback_bits = []
        if state.summary:
            fallback_bits.append(state.summary)
        for message in old_messages:
            content = str(message.get("content", "")).strip()
            if content:
                fallback_bits.append(content)
        state.summary = " | ".join(fallback_bits)[-1200:]

    state.messages = state.messages[-preserve_count:]


def submit_user_message(
    client: OpenAI, state: ConversationState, user_input: str, model: str
) -> str:
    state.messages.append({"role": "user", "content": user_input})

    try:
        answer = ask_chatbot(client, state, model)
    except Exception:
        state.messages.pop()
        raise

    state.messages.append({"role": "assistant", "content": answer})
    update_progress(state.progress, user_input)
    trim_memory(client, state, model)
    return answer
