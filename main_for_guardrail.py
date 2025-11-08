from __future__ import annotations

import os
import sys
import time
import atexit
import signal
import pandas as pd
from typing import Optional

from llama_cpp import Llama
from guardrails_medical import MedicalResponsePolicy, enforce_medical_policy


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
MODEL_PATH = r"C:\Users\shri\Documents\Apiphany\SLM\merged-Q8_0 (1).gguf"
CSV_LOG = "gemma1bmerged-Q4_0.csv"
QUESTION_FILE = "final_sensible_contextual_diabetes_batches_2.csv"  # used in Option 11
HISTORY_WINDOW = 2


# ------------------------------------------------------------
# LOAD MODEL (once)
# ------------------------------------------------------------
print("🔧 Loading model, please wait...")
llm = Llama(
    model_path=MODEL_PATH,
    n_threads=6,          # Adjust based on your CPU
    n_batch=128,
    verbose=False
)
print("✅ Model loaded successfully!\n")


# ------------------------------------------------------------
# INITIALIZE GUARDRAILS VALIDATOR
# ------------------------------------------------------------
medical_guard = MedicalResponsePolicy()
REFUSAL_LINE = MedicalResponsePolicy.REFUSAL_LINE  # mandated refusal message


# ------------------------------------------------------------
# CONVERSATION CONTEXT
# ------------------------------------------------------------
previous_question: Optional[str] = None
previous_response: Optional[str] = None
previous_passed: bool = False
conversation_history: list[tuple[str, str]] = []


# ------------------------------------------------------------
# CSV LOG SETUP (read existing; we will append in-memory and write once)
# ------------------------------------------------------------
columns = [
    "question",
    "response",
    "raw_model_response",
    "guardrails_status",
    "guardrail_reason",
    "first_token_time_sec",
    "tokens_per_sec",
    "total_time_sec",
]

try:
    df = pd.read_csv(CSV_LOG)
except FileNotFoundError:
    df = pd.DataFrame(columns=columns)

# Ensure required columns exist and order them
for column in columns:
    if column not in df.columns:
        df[column] = None
df = df[columns]

# We'll collect rows here and flush to CSV once at the end
pending_rows: list[dict] = []


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def build_prompt(question: str) -> str:
    """Create the system+context prompt."""
    history_blocks: list[str] = []
    for past_question, past_answer in conversation_history[-HISTORY_WINDOW:]:
        history_blocks.append(
            "### Previous Question:\n"
            f"{past_question.strip()}\n"
            "### Previous Answer:\n"
            f"{past_answer.strip()}\n"
        )
    history_section = "".join(history_blocks)
    if history_section:
        history_section += "\n"

    return (
        f"### Instruction:\n"
        "You are a medical chatbot with expert knowledge in diabetes management. "
        "Respond with medically accurate, empathetic information. "
        "Keep answers concise (2-4 sentences), avoid greetings, and reference prior context when helpful. "
        f"If the patient's question is unrelated to diabetes or human healthcare, respond exactly with: '{REFUSAL_LINE}'.\n\n"
        f"{history_section}"
        f"### Patient's Question:\n{question}\n\n"
        f"### Answer:"
    )


def answer_one_question(question: str) -> None:
    """Run the model + guardrails pipeline for a single question and stage a log row (no disk write here)."""
    global previous_question, previous_response, previous_passed

    print(f"\n{'='*60}\n🧠 Question:\n{question}\n{'='*60}")
    prompt = build_prompt(question)

    start_time = time.time()
    first_token_time = None
    output_text = ""
    tokens_generated = 0

    print("🤖 Model: ", end="", flush=True)

    # Stream generation
    for out in llm(
        prompt,
        max_tokens=150,
        temperature=0.0,
        top_p=0.5,
        stop=["###", "User:"],
        stream=True
    ):
        token = out["choices"][0]["text"]
        if first_token_time is None:
            first_token_time = time.time() - start_time
        print(token, end="", flush=True)
        output_text += token
        tokens_generated += 1

    total_time = time.time() - start_time
    tokens_per_sec = tokens_generated / total_time if total_time > 0 else 0.0

    raw_model_response = output_text.strip()

    # Guardrails enforcement
    passed_guardrails, final_response, failure_message = enforce_medical_policy(
        output_text,
        question,
        validator=medical_guard,
        previous_question=previous_question,
        previous_response=previous_response,
        previous_passed=previous_passed,
    )
    guardrails_status = "pass" if passed_guardrails else "auto_refusal"
    guardrail_reason = failure_message if failure_message else None

    if not passed_guardrails:
        print(f"\n⚠️ Guardrails violation: {failure_message}")
        print(f"↪️ Applying refusal policy: {final_response}")

    # Stage the row in memory (no to_csv here)
    pending_rows.append({
        "question": question,
        "response": final_response,
        "raw_model_response": raw_model_response,
        "guardrails_status": guardrails_status,
        "guardrail_reason": guardrail_reason,
        "first_token_time_sec": round(first_token_time, 3) if first_token_time else None,
        "tokens_per_sec": round(tokens_per_sec, 2),
        "total_time_sec": round(total_time, 3),
    })

    print(f"\n🕒 First token: {first_token_time:.2f}s | ⏱️ Speed: {tokens_per_sec:.2f} tok/s | Total: {total_time:.2f}s")
    print("📝 Staged to memory.\n")

    # Update conversation memory
    previous_question = question
    previous_response = final_response if passed_guardrails else None
    previous_passed = passed_guardrails

    if passed_guardrails:
        conversation_history.append((question, final_response.strip()))
        if len(conversation_history) > HISTORY_WINDOW:
            del conversation_history[: len(conversation_history) - HISTORY_WINDOW]


def flush_logs(force: bool = False):
    """
    Write staged rows to CSV once.
    If force=True, ensure the file exists (header-only if no rows).
    """
    global df, pending_rows
    wrote = False

    if pending_rows:
        df = pd.concat([df, pd.DataFrame(pending_rows)], ignore_index=True)
        pending_rows.clear()
        df.to_csv(CSV_LOG, index=False)
        print(f"📊 Wrote rows to '{CSV_LOG}'.")
        wrote = True
    elif force:
        # Ensure the CSV exists even if no questions were asked
        if not os.path.exists(CSV_LOG):
            pd.DataFrame(columns=df.columns).to_csv(CSV_LOG, index=False)
            print(f"📄 Created header-only CSV at '{CSV_LOG}'.")
            wrote = True

    if not wrote and force:
        # File existed and no new rows; still confirm we’re done
        print(f"📁 No new rows. CSV is up to date at '{CSV_LOG}'.")


# Always flush on normal interpreter exit
atexit.register(lambda: flush_logs(force=True))


# Also catch Ctrl+C to flush immediately, then exit cleanly
def _sigint_handler(signum, frame):
    print("\n🛑 Caught KeyboardInterrupt. Flushing logs…")
    flush_logs(force=True)
    # 130 is conventional exit code for SIGINT
    try:
        sys.exit(130)
    except SystemExit:
        # In some environments exit may be intercepted; ensure no further code runs
        raise


signal.signal(signal.SIGINT, _sigint_handler)


# ------------------------------------------------------------
# MODES
# ------------------------------------------------------------
def run_batch_from_file():
    """Option 1: load questions from QUESTION_FILE and process all, then flush once."""
    try:
        with open(QUESTION_FILE, "r", encoding="utf-8") as f:
            questions = [q.strip() for q in f.readlines() if q.strip()]
    except FileNotFoundError:
        print(f"❌ Could not find '{QUESTION_FILE}'. Create it or change QUESTION_FILE path.")
        return

    print(f"📄 Loaded {len(questions)} questions from '{QUESTION_FILE}'.")
    for i, q in enumerate(questions, start=1):
        print(f"\n➡️ Processing {i}/{len(questions)}")
        answer_one_question(q)

    flush_logs(force=True)
    print(f"\n✅ All {len(questions)} questions processed and logged.")


def run_interactive():
    """Option 2: ask custom questions from the terminal; flush once when you exit or on interrupt/EOF."""
    print("📝 Type your question (or 'exit' to quit):")
    try:
        while True:
            try:
                q = input("You: ").strip()
            except EOFError:
                print("\n📥 EOF received. Exiting…")
                break

            if q.lower() in {"exit", "quit"}:
                break
            if not q:
                continue

            answer_one_question(q)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")
    finally:
        flush_logs(force=True)
        print("👋 Bye!")


def main_menu():
    print("==============================================")
    print(" Choose an option:")
    print("  1) Ask questions from file")
    print("  2) Ask custom questions (interactive)")
    print("==============================================")
    choice = input("Enter 1 or 2: ").strip()
    if choice == "1":
        run_batch_from_file()
    elif choice == "2":
        run_interactive()
    else:
        print("❌ Invalid choice. Please run again and enter 1 or 2.")


# ------------------------------------------------------------
# ENTRYPOINT
# ------------------------------------------------------------
if __name__ == "__main__":
    main_menu()
