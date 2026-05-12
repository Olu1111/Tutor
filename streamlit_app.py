import os
import json
from typing import Any, Dict, List
from pathlib import Path

import streamlit as st

from code_evaluator import evaluate_submission, get_default_exercises
from code_evaluator import EvaluationCase, QuizExercise
from conversation import (
    DEFAULT_MODEL,
    build_client,
    load_api_key_text,
    new_conversation,
    record_usage_from_response,
    submit_user_message,
)

PRACTICE_TRIGGER_TERMS = (
    "practice question",
    "practice questions",
    "practice",
    "give me practice",
    "practice problem",
    "practice problems",
    "coding practice",
    "coding question",
    "coding questions",
    "code question",
    "code questions",
    "python question",
    "python questions",
    "coding exercise",
    "coding exercises",
    "code challenge",
    "code challenges",
    "code challenge",
    "quiz",
    "quiz me",
    "give me a quiz",
)


def is_practice_request(text: str) -> bool:
    lowered = text.lower()
    if any(term in lowered for term in PRACTICE_TRIGGER_TERMS):
        return True

    has_question_intent = any(word in lowered for word in ("question", "questions", "problem", "problems", "exercise", "exercises", "challenge"))
    has_code_intent = any(word in lowered for word in ("code", "coding", "python", "function", "implement"))
    return has_question_intent and has_code_intent


def append_message(role: str, content: str) -> None:
    st.session_state.messages.messages.append({"role": role, "content": content})


def _strip_json_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


def _exercise_from_spec(spec: Dict[str, Any]) -> QuizExercise:
    cases: List[EvaluationCase] = []
    for case in spec.get("cases", []):
        args = tuple(case.get("args", []))
        kwargs = case.get("kwargs", {})
        expected = case.get("expected", None)
        expected_stdout = case.get("expected_stdout", "")
        cases.append(
            EvaluationCase(
                args=args,
                kwargs=kwargs,
                expected=expected,
                expected_stdout=expected_stdout,
            )
        )

    return QuizExercise(
        name=spec.get("name", "Coding Question"),
        prompt=spec.get("prompt", "Solve the coding prompt."),
        function_name=spec.get("function_name", "solve"),
        starter_code=spec.get(
            "starter_code",
            f"def {spec.get('function_name', 'solve')}(...):\n    pass\n",
        ),
        cases=tuple(cases),
    )


def generate_evaluator_spec_from_question(client, model: str, question_text: str) -> QuizExercise:
    spec_prompt = (
        "Create an evaluation spec for a Python coding question. "
        "Return strict JSON only (no markdown). "
        "Include keys: name, prompt, function_name, starter_code, cases. "
        "Each item in cases must include args (JSON list), kwargs (JSON object), expected, expected_stdout (string). "
        "Use 3-6 deterministic test cases and avoid imports.\n\n"
        f"Question:\n{question_text}"
    )

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": "You generate safe, deterministic Python unit-test specs in valid JSON.",
            },
            {"role": "user", "content": spec_prompt},
        ],
    )
    record_usage_from_response(st.session_state.messages.usage, model, response)
    raw = _strip_json_fence(response.output_text)
    data = json.loads(raw)
    exercise = _exercise_from_spec(data)
    if not exercise.cases:
        raise ValueError("Generated evaluator spec did not include test cases.")
    return exercise


def start_next_practice_question(client, model: str, user_request: str) -> None:
    request = (
        "Generate one Python coding practice question for this student request: "
        f"{user_request}\n"
        "Question should be solvable with one function and have clear inputs/outputs."
    )
    exercise = generate_evaluator_spec_from_question(client, model, request)

    st.session_state.practice_mode = True
    st.session_state.current_exercise = exercise
    st.session_state.practice_code = exercise.starter_code
    st.session_state.messages.progress.questions_asked += 1

    append_message(
        "assistant",
        (
            f"Practice Question {st.session_state.messages.progress.questions_asked}: {exercise.prompt}\n\n"
            f"Implement `{exercise.function_name}` in the code box below, then press **Evaluate code**."
        ),
    )


def start_fallback_practice_question(fallback_exercises, user_request: str) -> None:
    lowered = user_request.lower()
    chosen = fallback_exercises[0]
    for exercise in fallback_exercises:
        if exercise.name.lower() in lowered:
            chosen = exercise
            break

    st.session_state.practice_mode = True
    st.session_state.current_exercise = chosen
    st.session_state.practice_code = chosen.starter_code
    st.session_state.messages.progress.questions_asked += 1
    append_message(
        "assistant",
        (
            f"Practice Question {st.session_state.messages.progress.questions_asked}: {chosen.prompt}\n\n"
            f"Implement `{chosen.function_name}` in the code box below, then press **Evaluate code**."
        ),
    )


def count_errors_from_report(report) -> int:
    if report.syntax_error or report.blocked_reason or report.timed_out:
        return 1
    failed_cases = sum(1 for case in report.case_results if not case.passed)
    return max(1, failed_cases)


def get_code_fix_suggestions(client, model: str, exercise, student_code: str, report) -> str:
    details = [f"Result: {report.summary()}"]
    for case in report.case_results:
        if case.passed:
            continue
        detail = f"Case {case.case_index} failed."
        if case.error:
            detail += f" Runtime error: {case.error}."
        detail += f" Expected {case.expected}, got {case.actual}."
        if case.stdout:
            detail += f" Stdout was {case.stdout}."
        details.append(detail)

    feedback_prompt = (
        "A student attempted this practice exercise and failed some tests. "
        "Give concise, actionable hints (bullets), explain likely bug patterns, and suggest what to check next. "
        "Do not provide the full final solution.\n\n"
        f"Exercise: {exercise.prompt}\n"
        f"Target function: {exercise.function_name}\n\n"
        f"Student code:\n{student_code}\n\n"
        + "\n".join(details)
    )

    try:
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": "You are a Python tutor. Give hints, not full solutions.",
                },
                {"role": "user", "content": feedback_prompt},
            ],
        )
        record_usage_from_response(st.session_state.messages.usage, model, response)
        suggestion = response.output_text.strip()
        return suggestion or "Check your base case, return values, and off-by-one logic, then try again."
    except Exception:
        return "Check your base case, return values, and expected output format, then try again."

st.set_page_config(page_title="Tutor Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 Tutor Chatbot")
st.caption("Ask Python questions and get guided explanations, examples, debugging help, and code evaluation.")

fallback_exercises = list(get_default_exercises())

if "messages" not in st.session_state:
    st.session_state.messages = new_conversation()

if "client" not in st.session_state:
    api_key = load_api_key_text(Path(__file__).with_name("tutor_key.txt"))
    st.session_state.client = build_client(api_key)

if "practice_mode" not in st.session_state:
    st.session_state.practice_mode = False

if "practice_code" not in st.session_state:
    st.session_state.practice_code = ""

if "current_exercise" not in st.session_state:
    st.session_state.current_exercise = fallback_exercises[0]

with st.sidebar:
    st.header("Settings")
    model = st.text_input("Model", value=os.getenv("OPENAI_MODEL", DEFAULT_MODEL))
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = new_conversation()
        st.session_state.practice_mode = False
        st.session_state.practice_code = ""
        st.session_state.current_exercise = fallback_exercises[0]
        st.rerun()

    st.subheader("Student progress")
    progress = st.session_state.messages.progress
    st.metric("Questions", progress.questions_asked)
    st.metric("Current topic", progress.current_topic or "None")
    st.metric("Confirms", progress.confirmations)
    st.metric("Struggles", progress.struggle_signals)

    with st.expander("Progress summary"):
        st.write(progress.to_summary())
    with st.expander("Conversation memory"):
        st.write(st.session_state.messages.summary or "No long-term memory yet.")

    with st.expander("Token + cost analysis"):
        usage = st.session_state.messages.usage
        st.metric("Model calls", usage.call_count)
        st.metric("Input tokens", usage.total_input_tokens)
        st.metric("Output tokens", usage.total_output_tokens)
        st.metric("Estimated cost (USD)", f"${usage.total_estimated_cost_usd:.6f}")
        st.caption("Cost is estimated from model pricing table in conversation.py and usage metadata returned by the API.")

for message in st.session_state.messages.messages:
    if message["role"] == "system":
        continue

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask a Python question...")
if prompt:
    if is_practice_request(prompt):
        append_message("user", prompt)
        try:
            start_next_practice_question(st.session_state.client, model, prompt)
        except Exception as error:
            append_message(
                "assistant",
                (
                    "I couldn't generate a dynamic evaluator for that request, "
                    "so I opened a fallback practice evaluator instead. "
                    f"(detail: {error})"
                ),
            )
            start_fallback_practice_question(fallback_exercises, prompt)
        st.rerun()

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Tutor is thinking..."):
            try:
                answer = submit_user_message(
                    st.session_state.client,
                    st.session_state.messages,
                    prompt,
                    model,
                )
            except Exception as error:
                answer = None
                st.error(f"API error: {error}")

        if answer:
            st.markdown(answer)

if st.session_state.practice_mode:
    current_exercise = st.session_state.current_exercise
    st.divider()
    st.subheader("Practice evaluator")
    st.caption("Use this area to submit code for the current coding question and retry until it passes.")

    with st.expander("Need a different coding question?"):
        custom_question_request = st.text_input(
            "Describe the coding question you want",
            placeholder="Example: Give me a medium recursion question",
        )
        if st.button("Generate question + evaluator", use_container_width=True):
            if custom_question_request.strip():
                try:
                    start_next_practice_question(
                        st.session_state.client,
                        model,
                        custom_question_request.strip(),
                    )
                    st.rerun()
                except Exception as error:
                    st.error(f"Could not generate evaluator from this request: {error}")
            else:
                st.warning("Enter a question request first.")

    if st.button("Show starter code", use_container_width=True):
        st.session_state.practice_code = current_exercise.starter_code

    student_code = st.text_area(
        "Your coding answer",
        value=st.session_state.practice_code,
        height=320,
        key="practice_code_editor",
    )
    st.session_state.practice_code = student_code

    if st.button("Evaluate code", type="primary", use_container_width=True):
        with st.spinner("Running evaluator..."):
            report = evaluate_submission(student_code, current_exercise)

        st.metric("Evaluation", report.summary())

        if report.success:
            st.session_state.messages.progress.confirmations += 1
            st.session_state.messages.progress.struggle_signals = 0

            append_message(
                "assistant",
                "Great job — your code passed all tests. Ask for another practice question whenever you're ready.",
            )
            st.session_state.practice_mode = False
            st.rerun()

        error_count = count_errors_from_report(report)
        st.session_state.messages.progress.struggle_signals += error_count

        suggestions = get_code_fix_suggestions(
            st.session_state.client,
            model,
            current_exercise,
            student_code,
            report,
        )

        append_message(
            "assistant",
            (
                f"Not quite yet. I found **{error_count}** issue(s).\n\n"
                f"Suggestions to fix your code:\n{suggestions}\n\n"
                "Update your code and evaluate again."
            ),
        )
        st.rerun()
