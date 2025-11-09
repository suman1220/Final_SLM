from __future__ import annotations

import atexit
import csv
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from typing import Iterable, Optional

from llama_cpp import Llama
from guardrails_medical import MedicalResponsePolicy, enforce_medical_policy
from guardrails_security import (
    crisis_response,
    detect_high_risk_medical_request,
    detect_prompt_injection,
    detect_self_harm_crisis,
    enforce_safety_filters,
)


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
MODEL_PATH = Path("/home/labs/SLM_UI/model.gguf")
CSV_LOG_PATH = Path("merged-Q8_0 (1).csv")
QUESTION_FILE = Path("question.txt")  # used in Option 1
HISTORY_WINDOW = 2
CACHE_PATH = Path("response_cache.json")
FEEDBACK_PATH = Path("response_feedback.json")
CPU_THREADS = max(1, (os.cpu_count() or 1))
RETRAINING_LOCK_PATH = Path("retraining.lock")


# ------------------------------------------------------------
# LOAD MODEL (once)
# ------------------------------------------------------------
print("Loading model, please wait...")
llm = Llama(
    model_path=str(MODEL_PATH),
    n_threads=CPU_THREADS,
    n_batch=256,
    n_gpu_layers=0,
    use_mmap=True,
    verbose=False
)
print("Model loaded successfully!")

print("Warming up model for faster first response...")
try:
    llm(
        "Warm-up pass.",
        max_tokens=1,
        temperature=0.0,
        top_p=1.0,
        stream=False,
    )
    print("Warm-up complete.\n")
except Exception as exc:
    print(f"Warning: Warm-up skipped: {exc}\n")


# ------------------------------------------------------------
# INITIALIZE GUARDRAILS VALIDATOR
# ------------------------------------------------------------
medical_guard = MedicalResponsePolicy()
REFUSAL_LINE = MedicalResponsePolicy.REFUSAL_LINE  # mandated refusal message
PROMPT_HEADER = (
    "### Instruction:\n"
    "You are an expert clinical assistant focused on diabetes and closely related metabolic care. "
    "Respond with evidence-based, empathetic guidance that aligns with current standards of practice. "
    "Use no more than 40 words unless additional detail is essential for patient safety. "
    "Choose the clearest structure for the user—bulleted list, numbered steps, compact markdown table, or short paragraph—based on the question intent. "
    "Avoid greetings, avoid chit-chat, and reference prior conversation context when it improves clarity. "
    "If the patient's question is unrelated to diabetes or human healthcare, respond exactly with "
    f"'{REFUSAL_LINE}'.\n\n"
)


# ------------------------------------------------------------
# CONVERSATION CONTEXT
# ------------------------------------------------------------
previous_question: Optional[str] = None
previous_response: Optional[str] = None
previous_passed: bool = False
conversation_history: list[tuple[str, str]] = []
conversation_history_formatted: list[str] = []


# ------------------------------------------------------------
# CSV LOG SETUP (normalize columns once, append per response)
# ------------------------------------------------------------
LOG_COLUMNS = [
    "question",
    "response",
    "first_token_time_sec",
    "tokens_per_sec",
    "total_time_sec",
]


def _initialize_log_file() -> None:
    """Ensure the CSV log exists and only contains the expected columns."""
    if CSV_LOG_PATH.exists():
        try:
            existing = pd.read_csv(CSV_LOG_PATH)
        except pd.errors.EmptyDataError:
            existing = pd.DataFrame(columns=LOG_COLUMNS)
        for column in LOG_COLUMNS:
            if column not in existing.columns:
                existing[column] = None
        existing = existing[LOG_COLUMNS]
    else:
        existing = pd.DataFrame(columns=LOG_COLUMNS)

    existing.to_csv(CSV_LOG_PATH, index=False)


_initialize_log_file()

cache_path = CACHE_PATH
try:
    with cache_path.open("r", encoding="utf-8") as cache_file:
        raw_cache = json.load(cache_file)
except (FileNotFoundError, json.JSONDecodeError):
    raw_cache = {}


def _normalize_question(question: str) -> str:
    """Canonical form for cache keys."""
    return " ".join(question.lower().split())


def build_cache_key(
    question: str,
    *,
    include_pending_pair: tuple[str, str] | None = None,
) -> str:
    """Create a cache key that incorporates recent conversation context."""
    normalized_question = _normalize_question(question)
    context_pairs: list[str] = []

    history_slice = conversation_history[-HISTORY_WINDOW:]
    for past_question, past_answer in history_slice:
        context_pairs.append(
            f"{_normalize_question(past_question)}=>{_normalize_question(past_answer)}"
        )

    if include_pending_pair:
        pending_question, pending_answer = include_pending_pair
        context_pairs.append(
            f"{_normalize_question(pending_question)}=>{_normalize_question(pending_answer)}"
        )
        if len(context_pairs) > HISTORY_WINDOW:
            context_pairs = context_pairs[-HISTORY_WINDOW:]

    if not context_pairs:
        return normalized_question

    return f"{normalized_question}||ctx:{'||'.join(context_pairs)}"


response_cache: dict[str, dict] = {}
cache_dirty_on_load = False
for key, value in raw_cache.items():
    if not isinstance(value, dict):
        cache_dirty_on_load = True
        continue
    cached_response = value.get("response")
    if not cached_response or cached_response == REFUSAL_LINE:
        cache_dirty_on_load = True
        continue
    cache_key = key if isinstance(key, str) else str(key)
    if "||ctx:" not in cache_key and "||" not in cache_key:
        normalized = _normalize_question(cache_key)
        if normalized != cache_key:
            cache_dirty_on_load = True
        cache_key = normalized
    if any(
        timing_key in value
        for timing_key in ("first_token_time_sec", "tokens_per_sec", "total_time_sec")
    ):
        cache_dirty_on_load = True
    response_cache[cache_key] = {
        "question": value.get("question", ""),
        "response": cached_response,
    }


class RuntimeIOManager:
    """Buffer log and cache writes to minimize per-question filesystem latency."""

    def __init__(
        self,
        *,
        log_path: Path,
        cache_path: Path,
        cache_payload: dict[str, dict],
        log_flush_threshold: int = 20,
        cache_flush_threshold: int = 20,
    ) -> None:
        self._log_path = log_path
        self._cache_path = cache_path
        self._cache_payload = cache_payload
        self._log_flush_threshold = max(1, log_flush_threshold)
        self._cache_flush_threshold = max(1, cache_flush_threshold)

        self._log_buffer: list[dict] = []
        self._cache_dirty = False
        self._cache_ops_since_flush = 0

        atexit.register(self.flush_all)

    def append_log_row(self, row: dict) -> None:
        self._log_buffer.append(row)
        if len(self._log_buffer) >= self._log_flush_threshold:
            self.flush_logs()

    def store_cache_entry(self, key: str, payload: dict) -> None:
        self._cache_payload[key] = payload
        self._touch_cache_dirty()

    def delete_cache_entry(self, key: str) -> None:
        if key in self._cache_payload:
            self._cache_payload.pop(key)
            self._touch_cache_dirty()

    def flush_logs(self) -> None:
        if not self._log_buffer:
            return

        file_is_empty = not self._log_path.exists() or self._log_path.stat().st_size == 0
        with self._log_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=LOG_COLUMNS)
            if file_is_empty:
                writer.writeheader()
            writer.writerows(self._log_buffer)
        self._log_buffer.clear()

    def flush_cache(self) -> None:
        if not self._cache_dirty:
            return

        try:
            with self._cache_path.open("w", encoding="utf-8") as cache_file:
                json.dump(self._cache_payload, cache_file, ensure_ascii=False, indent=2)
        except OSError as exc:
            print(f"Warning: Unable to write cache file '{self._cache_path}': {exc}")
            return

        self._cache_dirty = False
        self._cache_ops_since_flush = 0

    def flush_all(self) -> None:
        self.flush_logs()
        self.flush_cache()

    def _touch_cache_dirty(self) -> None:
        self._cache_dirty = True
        self._cache_ops_since_flush += 1
        if self._cache_ops_since_flush >= self._cache_flush_threshold:
            self.flush_cache()


io_manager = RuntimeIOManager(
    log_path=CSV_LOG_PATH,
    cache_path=cache_path,
    cache_payload=response_cache,
)

if cache_dirty_on_load:
    io_manager.flush_cache()


def append_log_row(row: dict) -> None:
    """Append a single row to the CSV log via the buffered manager."""
    io_manager.append_log_row(row)


def store_cache_entry(
    cache_key: str,
    question: str,
    response: str,
    *,
    first_token_time: float,
    tokens_per_sec: float,
    total_time: float,
) -> None:
    """Persist a question/response pair for instant reuse."""
    io_manager.store_cache_entry(
        cache_key,
        {
            "question": question,
            "response": response,
        },
    )


def delete_cache_entry(cache_key: str) -> None:
    """Remove an entry from the in-memory cache and mark flush needed."""
    io_manager.delete_cache_entry(cache_key)


def flush_pending_io() -> None:
    """Ensure buffered cache and log data are flushed to disk."""
    io_manager.flush_all()
    feedback_store.flush()


class FeedbackStore:
    """Maintain good/bad feedback entries without impacting runtime latency."""

    _ALLOWED_KEYS = {"question", "response", "verdict", "timestamp"}

    def __init__(
        self,
        *,
        storage_path: Path,
        flush_threshold: int = 10,
    ) -> None:
        self._path = storage_path
        self._flush_threshold = max(1, flush_threshold)
        self._payload: dict[str, list[dict]] = {"good": [], "bad": []}
        self._buffer_count = 0
        self._dirty = False

        try:
            with self._path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError):
            data = None

        if isinstance(data, dict):
            for key in ("good", "bad"):
                bucket = data.get(key)
                if isinstance(bucket, list):
                    self._payload[key] = [
                        self._sanitize_entry(item)
                        for item in bucket
                        if isinstance(item, dict)
                    ]

        atexit.register(self.flush)

    def record(self, verdict: str, entry: dict) -> None:
        bucket = "good" if verdict == "good" else "bad"
        self._payload[bucket].append(self._sanitize_entry(entry))
        self._buffer_count += 1
        self._dirty = True

        if self._buffer_count >= self._flush_threshold:
            self.flush()

    def iter_entries(self, *, verdict: str | None = None) -> Iterable[dict]:
        if verdict == "good":
            yield from self._payload["good"]
            return
        if verdict == "bad":
            yield from self._payload["bad"]
            return
        yield from self._payload["good"]
        yield from self._payload["bad"]

    def flush(self) -> None:
        if not self._dirty:
            return

        try:
            with self._path.open("w", encoding="utf-8") as handle:
                json.dump(self._payload, handle, ensure_ascii=False, indent=2)
        except OSError as exc:
            print(f"Warning: Unable to write feedback file '{self._path}': {exc}")
            return

        self._dirty = False
        self._buffer_count = 0

    def _sanitize_entry(self, entry: dict) -> dict:
        return {
            key: entry[key]
            for key in self._ALLOWED_KEYS
            if key in entry
        }


feedback_store = FeedbackStore(storage_path=FEEDBACK_PATH)


def replay_feedback_into_cache(*, only_good: bool = True) -> int:
    """Rehydrate the runtime cache from stored feedback entries."""
    verdict = "good" if only_good else None
    restored = 0
    for entry in feedback_store.iter_entries(verdict=verdict):
        question = entry.get("question")
        response = entry.get("response")
        if not question or not response:
            continue
        cache_key = build_cache_key(question)
        store_cache_entry(
            cache_key,
            question,
            response,
            first_token_time=0.0,
            tokens_per_sec=0.0,
            total_time=0.0,
        )
        restored += 1
    return restored


def _prompt_for_feedback(
    question: str,
    response: str,
    metrics: dict,
) -> None:
    """Collect binary feedback without delaying first-token latency."""
    while True:
        choice = input("Feedback (y=good / n=bad / Enter=skip): ").strip().lower()
        if choice in {"", "y", "n"}:
            break
        print("Please respond with 'y', 'n', or press Enter to skip.")

    if not choice:
        return

    verdict = "good" if choice == "y" else "bad"
    entry = {
        "question": question,
        "response": response,
        "verdict": verdict,
        "timestamp": time.time(),
    }
    feedback_store.record(verdict, entry)


def _maybe_collect_feedback(
    *,
    enabled: bool,
    question: str,
    response: str,
    metrics: dict,
) -> None:
    if not enabled:
        return
    _prompt_for_feedback(question, response, metrics)


def retraining_in_progress() -> bool:
    return RETRAINING_LOCK_PATH.exists()


def _inform_retraining() -> None:
    print("We are updating the chatbot with new training. Please try again shortly.")


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def build_prompt(question: str) -> str:
    """Create the system+context prompt."""
    history_section = "".join(conversation_history_formatted[-HISTORY_WINDOW:])
    if history_section:
        history_section += "\n"

    return (
        f"{PROMPT_HEADER}"
        f"{history_section}"
        f"### Patient's Question:\n{question}\n\n"
        "### Answer:"
    )


def _append_conversation_entry(question: str, response: str) -> None:
    """Track recent conversation context with cached formatted blocks."""
    condensed_response = response.strip()
    conversation_history.append((question, condensed_response))
    conversation_history_formatted.append(
        "### Previous Question:\n"
        f"{question.strip()}\n"
        "### Previous Answer:\n"
        f"{condensed_response}\n"
    )
    if len(conversation_history) > HISTORY_WINDOW:
        excess = len(conversation_history) - HISTORY_WINDOW
        del conversation_history[:excess]
        del conversation_history_formatted[:excess]


def answer_one_question(question: str, *, request_feedback: bool = False) -> None:
    """Run the model + guardrails pipeline (with cache + immediate logging)."""
    global previous_question, previous_response, previous_passed

    if retraining_in_progress():
        _inform_retraining()
        return

    print(f"\n{'=' * 60}\nQuestion:\n{question}\n{'=' * 60}")
    cache_key = build_cache_key(question)

    injection_result = detect_prompt_injection(
        question,
        recent_questions=(q for q, _ in conversation_history),
    )
    if not injection_result.passed:
        if injection_result.message:
            print(f"Warning: {injection_result.message}")
        print(REFUSAL_LINE)
        row = {
            "question": question,
            "response": REFUSAL_LINE,
            "first_token_time_sec": 0.0,
            "tokens_per_sec": 0.0,
            "total_time_sec": 0.0,
        }
        append_log_row(row)

        previous_question = question
        previous_response = None
        previous_passed = False

        if cache_key in response_cache:
            delete_cache_entry(cache_key)
        print("First token: 0.00s | Speed: instant | Total: 0.00s")
        return

    crisis_result = detect_self_harm_crisis(question)
    if not crisis_result.passed:
        crisis_message = crisis_response()
        if crisis_result.message:
            print(f"Warning: {crisis_result.message}")
        print(crisis_message)
        row = {
            "question": question,
            "response": crisis_message,
            "first_token_time_sec": 0.0,
            "tokens_per_sec": 0.0,
            "total_time_sec": 0.0,
        }
        append_log_row(row)

        previous_question = question
        previous_response = None
        previous_passed = False

        if cache_key in response_cache:
            delete_cache_entry(cache_key)
        print("First token: 0.00s | Speed: instant | Total: 0.00s")
        return

    high_risk_result = detect_high_risk_medical_request(question)
    if not high_risk_result.passed:
        if high_risk_result.message:
            print(f"Warning: {high_risk_result.message}")
        print(REFUSAL_LINE)
        row = {
            "question": question,
            "response": REFUSAL_LINE,
            "first_token_time_sec": 0.0,
            "tokens_per_sec": 0.0,
            "total_time_sec": 0.0,
        }
        append_log_row(row)

        previous_question = question
        previous_response = None
        previous_passed = False

        if cache_key in response_cache:
            delete_cache_entry(cache_key)
        print("First token: 0.00s | Speed: instant | Total: 0.00s")
        return

    is_follow_up = medical_guard._looks_like_follow_up(
        question, previous_question, previous_response
    )
    if medical_guard._is_off_topic_question(question) and not is_follow_up:
        row = {
            "question": question,
            "response": REFUSAL_LINE,
            "first_token_time_sec": 0.0,
            "tokens_per_sec": 0.0,
            "total_time_sec": 0.0,
        }
        append_log_row(row)
        print(REFUSAL_LINE)
        print("First token: 0.00s | Speed: instant | Total: 0.00s")

        previous_question = question
        previous_response = None
        previous_passed = False

        if cache_key in response_cache:
            delete_cache_entry(cache_key)

        return

    fetch_start = time.time()
    cached_entry = response_cache.get(cache_key)
    fetch_duration = time.time() - fetch_start if cached_entry else 0.0

    if cached_entry:
        response_text = cached_entry.get("response", "")

        print("Cache hit - reusing stored answer.\n")
        print(response_text)

        row = {
            "question": question,
            "response": response_text,
            "first_token_time_sec": round(fetch_duration, 3),
            "tokens_per_sec": 0.0,
            "total_time_sec": round(fetch_duration, 3),
        }
        append_log_row(row)

        previous_question = question
        previous_response = response_text
        previous_passed = bool(response_text)

        if response_text:
            _append_conversation_entry(question, response_text)

        print(
            f"Cache fetch: {fetch_duration:.3f}s | Speed: cached | Total: {fetch_duration:.3f}s"
        )
        if response_text:
            _maybe_collect_feedback(
                enabled=request_feedback,
                question=question,
                response=response_text,
                metrics=row,
            )
        return

    prompt = build_prompt(question)

    start_time = time.time()
    first_token_time: Optional[float] = None
    output_text = ""
    tokens_generated = 0
    first_token_flushed = False

    sys.stdout.write("Model: ")
    sys.stdout.flush()

    for out in llm(
        prompt,
        max_tokens=150,
        temperature=0.0,
        top_p=0.5,
        stop=["###", "User:"],
        stream=True,
    ):
        token = out["choices"][0]["text"]
        if first_token_time is None:
            first_token_time = time.time() - start_time
        output_text += token
        tokens_generated += 1
        sys.stdout.write(token)
        if not first_token_flushed:
            sys.stdout.flush()
            first_token_flushed = True

    sys.stdout.write("\n")
    sys.stdout.flush()

    total_time = time.time() - start_time
    tokens_per_sec = tokens_generated / total_time if total_time > 0 else 0.0
    model_response = output_text.strip()

    passed_guardrails, final_response, failure_message = enforce_medical_policy(
        model_response,
        question,
        validator=medical_guard,
        previous_question=previous_question,
        previous_response=previous_response,
        previous_passed=previous_passed,
    )

    if passed_guardrails:
        safety_result = enforce_safety_filters(final_response)
        if not safety_result.passed:
            if safety_result.message:
                print(f"Warning: Safety filter violation: {safety_result.message}")
            passed_guardrails = False
            failure_message = safety_result.message
            final_response = REFUSAL_LINE

    if not passed_guardrails and failure_message:
        print(f"Warning: Guardrails violation: {failure_message}")
        print(f"Applying refusal policy: {final_response}")

    safe_response = final_response.strip()
    first_token_time = first_token_time or 0.0

    row = {
        "question": question,
        "response": safe_response,
        "first_token_time_sec": round(first_token_time, 3),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "total_time_sec": round(total_time, 3),
    }
    append_log_row(row)

    if passed_guardrails:
        store_cache_entry(
            cache_key,
            question,
            safe_response,
            first_token_time=row["first_token_time_sec"],
            tokens_per_sec=row["tokens_per_sec"],
            total_time=row["total_time_sec"],
        )

        future_cache_key = build_cache_key(
            question,
            include_pending_pair=(question, safe_response),
        )
        if future_cache_key != cache_key:
            store_cache_entry(
                future_cache_key,
                question,
                safe_response,
                first_token_time=row["first_token_time_sec"],
                tokens_per_sec=row["tokens_per_sec"],
                total_time=row["total_time_sec"],
            )

    print(
        f"First token: {first_token_time:.2f}s | Speed: {tokens_per_sec:.2f} tok/s | Total: {total_time:.2f}s"
    )

    previous_question = question
    previous_response = safe_response if passed_guardrails else None
    previous_passed = passed_guardrails

    if passed_guardrails:
        _append_conversation_entry(question, safe_response)

    if passed_guardrails:
        _maybe_collect_feedback(
            enabled=request_feedback,
            question=question,
            response=safe_response,
            metrics=row,
        )


# ------------------------------------------------------------
# MODES
# ------------------------------------------------------------
def run_batch_from_file():
    """Option 1: load questions from QUESTION_FILE and process all."""
    if retraining_in_progress():
        _inform_retraining()
        return

    try:
        with QUESTION_FILE.open("r", encoding="utf-8") as f:
            questions = [q.strip() for q in f.readlines() if q.strip()]
    except FileNotFoundError:
        print(f"Could not find '{QUESTION_FILE}'. Create it or change QUESTION_FILE path.")
        return

    print(f"Loaded {len(questions)} questions from '{QUESTION_FILE}'.")
    for i, q in enumerate(questions, start=1):
        print(f"\nProcessing {i}/{len(questions)}")
        answer_one_question(q)

    print(f"\nAll {len(questions)} questions processed and logged.")
    flush_pending_io()


def run_interactive():
    """Option 2: ask custom questions from the terminal."""
    print("Type your question (or 'exit' to quit):")
    if retraining_in_progress():
        _inform_retraining()
        print("You can wait here or type a question to check again.")
    try:
        while True:
            try:
                q = input("You: ").strip()
            except EOFError:
                print("\nEOF received. Exiting…")
                break

            if q.lower() in {"exit", "quit"}:
                break
            if not q:
                continue
            if retraining_in_progress():
                _inform_retraining()
                continue

            answer_one_question(q, request_feedback=True)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        flush_pending_io()
        print("Bye!")


# ------------------------------------------------------------
# ENTRYPOINT
# ------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in {"--batch", "-b"}:
        run_batch_from_file()
    else:
        run_interactive()
