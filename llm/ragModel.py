from __future__ import annotations

import os
import sys
import time
import atexit
import signal
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd

from llama_cpp import Llama
from guardrails_medical import MedicalResponsePolicy, enforce_medical_policy

# NEW: import the RAG module (same folder)
from rag_engine import RAGEngine, RAGConfig

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
MODEL_PATH = r"Vineet's_v2gemma1bmerged-Q4_0.gguf"
CSV_LOG = "gemma1bmerged-Q4_0.csv"
QUESTION_FILE = "question.txt"  # used in Option 1
HISTORY_WINDOW = 2

# RAG config (tweak as needed)
RAG_DOCS_DIR = "./data/documents"
rag_cfg = RAGConfig(
    docs_dir=RAG_DOCS_DIR,
    chunk_size=300,
    chunk_overlap=60,
    sents_per_chunk=1,
    context_budget=150,
    k=4,
    bm25_fallback=True,
    rag_lex_min=2,
)
RAG_ENABLE = True  # master switch

# ------------------------------------------------------------
# LOAD MODEL (once)
# ------------------------------------------------------------
print("🔧 Loading model, please wait...")
llm = Llama(
    model_path=MODEL_PATH,
    n_threads=6,
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
# CSV LOG SETUP
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
    "rag_mode",
    "rag_sources",
]
try:
    df = pd.read_csv(CSV_LOG)
except FileNotFoundError:
    df = pd.DataFrame(columns=columns)
for c in columns:
    if c not in df.columns:
        df[c] = None
df = df[columns]
pending_rows: list[dict] = []

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def token_len(s: str) -> int:
    try:
        return len(llm.tokenize(s.encode("utf-8"), add_bos=False))
    except Exception:
        return max(1, int(len(s) * 0.26))

def _generate_with_llama(prompt: str) -> Tuple[str, float, float, float, int]:
    start_time = time.time()
    first_token_time = None
    output_text = ""
    tokens_generated = 0

    print("🤖 Model: ", end="", flush=True)
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
    return output_text.strip(), first_token_time or 0.0, tokens_per_sec, total_time, tokens_generated

def _history_section() -> str:
    blocks: List[str] = []
    for q, a in conversation_history[-HISTORY_WINDOW:]:
        blocks.append(
            "### Previous Question:\n"
            f"{q.strip()}\n"
            "### Previous Answer:\n"
            f"{a.strip()}\n"
        )
    hs = "".join(blocks)
    return (hs + "\n") if hs else ""

def build_prompt(question: str) -> str:
    return (
        f"### Instruction:\n"
        "You are a medical chatbot with expert knowledge in diabetes management. "
        "Respond with medically accurate, empathetic information. "
        "Keep answers concise (2-4 sentences), avoid greetings, and reference prior context when helpful. "
        f"If the patient's question is unrelated to diabetes or human healthcare, respond exactly with: '{REFUSAL_LINE}'.\n\n"
        f"{_history_section()}"
        f"### Patient's Question:\n{question}\n\n"
        f"### Answer:"
    )

def build_prompt_with_context(question: str, context: str) -> str:
    return (
        f"### Instruction:\n"
        "You are a medical chatbot with expert knowledge in diabetes management. "
        "Respond with medically accurate, empathetic information. "
        "Keep answers concise (2-4 sentences), avoid greetings, and reference prior context when helpful. "
        f"If the patient's question is unrelated to diabetes or human healthcare, respond exactly with: '{REFUSAL_LINE}'.\n\n"
        f"{_history_section()}"
        f"### Context (from documents):\n{context}\n\n"
        f"### Patient's Question:\n{question}\n\n"
        f"### Answer:"
    )

def _stage_row(question: str, final_response: str, raw_model_response: str,
               first_token_time: float, tokens_per_sec: float, total_time: float,
               rag_mode: str = "off", rag_sources: Optional[str] = None):
    pending_rows.append({
        "question": question,
        "response": final_response,
        "raw_model_response": raw_model_response,
        "guardrails_status": "pass",
        "guardrail_reason": None,
        "first_token_time_sec": round(first_token_time, 3) if first_token_time else None,
        "tokens_per_sec": round(tokens_per_sec, 2),
        "total_time_sec": round(total_time, 3),
        "rag_mode": rag_mode,
        "rag_sources": rag_sources,
    })

def _apply_guardrails_and_log(question: str, raw: str,
                              timings: Tuple[float, float, float],
                              rag_mode: str = "off", rag_sources: Optional[str] = None):
    global previous_question, previous_response, previous_passed
    first_token_time, tokens_per_sec, total_time = timings

    passed_guardrails, final_response, failure_message = enforce_medical_policy(
        raw, question,
        validator=medical_guard,
        previous_question=previous_question,
        previous_response=previous_response,
        previous_passed=previous_passed,
    )
    if not passed_guardrails:
        print(f"\n⚠️ Guardrails violation: {failure_message}")
        print(f"↪️ Applying refusal policy: {final_response}")

    _stage_row(
        question=question,
        final_response=final_response,
        raw_model_response=raw,
        first_token_time=first_token_time,
        tokens_per_sec=tokens_per_sec,
        total_time=total_time,
        rag_mode=("on" if rag_mode == "on" else "off"),
        rag_sources=rag_sources if rag_mode == "on" else None
    )

    print(f"\n🕒 First token: {first_token_time:.2f}s | ⏱️ Speed: {tokens_per_sec:.2f} tok/s | Total: {total_time:.2f}s")
    print("📝 Staged to memory.\n")

    previous_question = question
    previous_response = final_response if passed_guardrails else None
    previous_passed = passed_guardrails

    if passed_guardrails:
        conversation_history.append((question, final_response.strip()))
        if len(conversation_history) > HISTORY_WINDOW:
            del conversation_history[: len(conversation_history) - HISTORY_WINDOW]

# ------------------------------------------------------------
# CSV FLUSH & SIGNALS
# ------------------------------------------------------------
def flush_logs(force: bool = False):
    global df, pending_rows
    wrote = False
    if pending_rows:
        df = pd.concat([df, pd.DataFrame(pending_rows)], ignore_index=True)
        pending_rows.clear()
        df.to_csv(CSV_LOG, index=False)
        print(f"📊 Wrote rows to '{CSV_LOG}'.")
        wrote = True
    elif force:
        if not os.path.exists(CSV_LOG):
            pd.DataFrame(columns=df.columns).to_csv(CSV_LOG, index=False)
            print(f"📄 Created header-only CSV at '{CSV_LOG}'.")
            wrote = True
    if not wrote and force:
        print(f"📁 No new rows. CSV is up to date at '{CSV_LOG}'.")

import atexit as _atexit
_atexit.register(lambda: flush_logs(force=True))

def _sigint_handler(signum, frame):
    print("\n🛑 Caught KeyboardInterrupt. Flushing logs…")
    flush_logs(force=True)
    try:
        sys.exit(130)
    except SystemExit:
        raise

signal.signal(signal.SIGINT, _sigint_handler)

# ------------------------------------------------------------
# CORE Q&A (Option 1 & 2: non-RAG)
# ------------------------------------------------------------
def answer_one_question(question: str) -> None:
    print(f"\n{'='*60}\n🧠 Question:\n{question}\n{'='*60}")
    prompt = build_prompt(question)
    raw, first_token_time, tps, total_time, _ = _generate_with_llama(prompt)
    _apply_guardrails_and_log(question, raw, (first_token_time, tps, total_time), rag_mode="off", rag_sources=None)

# ------------------------------------------------------------
# RAG (Option 3) using external module
# ------------------------------------------------------------
_rag_engine: Optional[RAGEngine] = None
_rag_indexed_count: int = 0

def ensure_rag_ready():
    global _rag_engine, _rag_indexed_count
    if not RAG_ENABLE:
        raise RuntimeError("RAG disabled by config.")
    if _rag_engine is None:
        _rag_engine = RAGEngine(rag_cfg, token_len_fn=token_len)
        os.makedirs(rag_cfg.docs_dir, exist_ok=True)
        try:
            _rag_indexed_count = _rag_engine.index_pdfs_in_dir(rag_cfg.docs_dir)
            if _rag_indexed_count == 0:
                print(f"⚠️ No PDFs found in '{rag_cfg.docs_dir}'. RAG will fall back to generic answers.")
        except Exception as e:
            print(f"❌ RAG indexing failed: {e}")
            _rag_indexed_count = 0
    return _rag_engine, _rag_indexed_count

def answer_one_question_rag(question: str) -> None:
    engine, count = ensure_rag_ready()
    print(f"\n{'='*60}\n📚 RAG Question:\n{question}\n{'='*60}")
    if count <= 0:
        # No docs -> generic
        return answer_one_question(question)

    hits, top_sim, cnt = engine.search_with_score(question, rag_cfg.k)
    if cnt == 0 or not hits or not engine.should_use_rag(question, hits):
        # Not enough signal -> generic
        prompt = build_prompt(question)
        raw, first_token_time, tps, total_time, _ = _generate_with_llama(prompt)
        _apply_guardrails_and_log(question, raw, (first_token_time, tps, total_time), rag_mode="off", rag_sources=None)
        return

    context, chosen_meta = engine.assemble_context(question, hits)
    rag_sources = []
    for m in chosen_meta:
        src = m.get("source", "?"); page = m.get("page", "?"); section = m.get("section") or ""
        rag_sources.append(f"{src} (p.{page}){(' · '+section) if section else ''}")
    rag_sources_str = " | ".join(rag_sources) if rag_sources else None
    if rag_sources_str:
        print(f"🔎 Using sources: {rag_sources_str}")

    prompt = build_prompt_with_context(question, context)
    raw, first_token_time, tps, total_time, _ = _generate_with_llama(prompt)
    _apply_guardrails_and_log(question, raw, (first_token_time, tps, total_time), rag_mode="on", rag_sources=rag_sources_str)

# ------------------------------------------------------------
# MODES
# ------------------------------------------------------------
def run_batch_from_file():
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

def run_rag_interactive():
    try:
        ensure_rag_ready()
    except Exception as e:
        print(f"❌ RAG init failed: {e}")
        return
    print("📝 RAG chat ready. Type your question (or 'exit' to quit):")
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
            answer_one_question_rag(q)
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
    print("  3) RAG: index PDFs and chat with retrieval")
    print("==============================================")
    choice = input("Enter 1, 2 or 3: ").strip()
    if choice == "1":
        run_batch_from_file()
    elif choice == "2":
        run_interactive()
    elif choice == "3":
        run_rag_interactive()
    else:
        print("❌ Invalid choice. Please run again and enter 1, 2 or 3.")

# ------------------------------------------------------------
# ENTRYPOINT
# ------------------------------------------------------------
if __name__ == "__main__":
    main_menu()