import os
from pathlib import Path

import streamlit as st

from code_evaluator import evaluate_submission, get_default_exercises
from conversation import (
    DEFAULT_MODEL,
    build_client,
    load_api_key_text,
    new_conversation,
    submit_user_message,
)

st.set_page_config(page_title="Tutor Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 Tutor Chatbot")
st.caption("Ask Python questions and get guided explanations, examples, and debugging help.")

exercises = list(get_default_exercises())

if "messages" not in st.session_state:
    st.session_state.messages = new_conversation()

if "client" not in st.session_state:
    api_key = load_api_key_text(Path(__file__).with_name("tutor_key.txt"))
    st.session_state.client = build_client(api_key)

with st.sidebar:
    st.header("Settings")
    model = st.text_input("Model", value=os.getenv("OPENAI_MODEL", DEFAULT_MODEL))
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = new_conversation()
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

chat_tab, evaluator_tab = st.tabs(["Chat", "Code evaluator"])

with chat_tab:
    for message in st.session_state.messages.messages:
        if message["role"] == "system":
            continue

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask a Python question...")
    if prompt:
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

with evaluator_tab:
    st.subheader("Auto-evaluate student code")
    st.write("Select a practice exercise, paste student code, and run safe checks against hidden tests.")

    exercise_names = [exercise.name for exercise in exercises]
    selected_name = st.selectbox("Exercise", exercise_names)
    selected_exercise = next(exercise for exercise in exercises if exercise.name == selected_name)

    if "student_code" not in st.session_state:
        st.session_state.student_code = selected_exercise.starter_code

    if st.button("Load starter code", use_container_width=True):
        st.session_state.student_code = selected_exercise.starter_code

    student_code = st.text_area(
        "Student code",
        key="student_code",
        height=320,
        help="This code is evaluated in a restricted subprocess with AST checks and a timeout.",
    )

    if st.button("Run auto-evaluator", type="primary", use_container_width=True):
        with st.spinner("Running safe checks..."):
            report = evaluate_submission(student_code, selected_exercise)

        st.metric("Result", report.summary())
        if report.blocked_reason:
            st.error(report.blocked_reason)
        elif report.syntax_error:
            st.error(report.syntax_error)
        elif report.timed_out:
            st.error("The submission timed out. Try simpler code or remove infinite recursion/loops.")
        else:
            st.success(report.summary())

        for case_result in report.case_results:
            status = "Passed" if case_result.passed else "Failed"
            with st.expander(f"Case {case_result.case_index}: {status}"):
                st.write(f"Expected: {case_result.expected}")
                if case_result.stdout:
                    st.write(f"Captured stdout: {case_result.stdout}")
                if case_result.actual:
                    st.write(f"Actual return value: {case_result.actual}")
                if case_result.error:
                    st.write(f"Error: {case_result.error}")

    with st.expander("Exercise prompt"):
        st.write(selected_exercise.prompt)
