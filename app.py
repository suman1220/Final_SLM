# app.py - FIXED VERSION matching llm/main.py behavior exactly
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional, Deque, Tuple
from flask import Flask, request, Response, jsonify, send_from_directory, render_template
from flask_cors import CORS
from llama_cpp import Llama
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add llm folder to path to import enhanced guardrails
sys.path.insert(0, str(Path(__file__).parent / "llm"))

from guardrails_medical import MedicalResponsePolicy, enforce_medical_policy
from guardrails_security import (
    crisis_response,
    detect_high_risk_medical_request,
    detect_prompt_injection,
    detect_self_harm_crisis,
    enforce_safety_filters,
)

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "model.gguf")
CPU_THREADS = max(1, (os.cpu_count() or 1))
N_THREADS = int(os.getenv("LLM_THREADS", str(CPU_THREADS)))
N_BATCH = int(os.getenv("LLM_BATCH", "256"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "150"))  # Match llm/main.py

app = Flask(__name__, template_folder="templates", static_folder="static", static_url_path="/static")
CORS(app)

# Load model with enhanced configuration matching llm/main.py
print(f"Loading model from {MODEL_PATH}...")
llm = Llama(
    model_path=MODEL_PATH,
    n_threads=N_THREADS,
    n_batch=N_BATCH,
    n_gpu_layers=0,
    use_mmap=True,
    verbose=False
)
print("Model loaded successfully")

# Warm up model for faster first response
print("Warming up model for faster first response...")
try:
    llm(
        "Warm-up pass.",
        max_tokens=1,
        temperature=0.0,
        top_p=1.0,
        stream=False,
    )
    print("Warm-up complete.")
except Exception as exc:
    print(f"Warning: Warm-up skipped: {exc}")

# Memory & Cache
HISTORY_WINDOW = 2
MEMORY_MAXLEN = 20
CACHE_TTL_SECONDS = 1800

medical_guard = MedicalResponsePolicy()
REFUSAL_LINE = MedicalResponsePolicy.REFUSAL_LINE

# Session-based conversation memory (matches llm/main.py global variables)
CONV_MEMORY: dict[str, Deque[Tuple[str, str, bool]]] = {}
QA_CACHE: dict[str, tuple[str, float]] = {}

def _normalize_q(q: str) -> str:
    return " ".join(q.split()).lower()

def _sid() -> str:
    return request.headers.get("X-Session-Id", "anon")

def _get_history_blocks(session_id: str) -> list[tuple[str, str]]:
    """Get conversation history blocks for building prompts"""
    dq = CONV_MEMORY.get(session_id)
    if not dq:
        return []
    hist = []
    for q, a, passed in list(dq)[-HISTORY_WINDOW:]:
        if passed and q and a:
            hist.append((q, a))
    return hist

def _get_conversation_history(session_id: str) -> list[tuple[str, str]]:
    """Get raw conversation history (for cache key building)"""
    dq = CONV_MEMORY.get(session_id)
    if not dq:
        return []
    return [(q, a) for q, a, passed in list(dq)[-HISTORY_WINDOW:] if passed and q and a]

def _remember(session_id: str, question: str, answer: str, passed: bool):
    """Add to conversation memory"""
    dq = CONV_MEMORY.setdefault(session_id, deque(maxlen=MEMORY_MAXLEN))
    dq.append((question, answer, passed))

def build_cache_key(session_id: str, question: str) -> str:
    """Build context-aware cache key matching llm/main.py build_cache_key() logic"""
    normalized_question = _normalize_q(question)

    history = _get_conversation_history(session_id)
    if not history:
        return f"{session_id}||{normalized_question}"

    # Build context pairs from conversation history
    context_pairs = []
    for past_q, past_a in history:
        context_pairs.append(f"{_normalize_q(past_q)}=>{_normalize_q(past_a)}")

    context_str = "||".join(context_pairs)
    return f"{session_id}||{normalized_question}||ctx:{context_str}"

def _get_cached(session_id: str, question: str):
    key = build_cache_key(session_id, question)
    v = QA_CACHE.get(key)
    if not v:
        return None
    ans, ts = v
    if (time.time() - ts) > CACHE_TTL_SECONDS:
        QA_CACHE.pop(key, None)
        return None
    return ans

def _put_cached(session_id: str, question: str, answer: str):
    key = build_cache_key(session_id, question)
    QA_CACHE[key] = (answer, time.time())

@app.get("/api/config")
def api_config():
    return jsonify({
        "model": os.path.basename(MODEL_PATH),
        "threads": N_THREADS,
        "batch": N_BATCH,
        "max_tokens": MAX_TOKENS
    })

@app.get("/api/health")
def health_check():
    return jsonify({"status": "healthy", "model_loaded": True}), 200

@app.get("/")
def root():
    return render_template("index.html")

def build_prompt_model_only(question: str) -> str:
    """Model-only prompt matching llm/main_only_model.py lines 47-56"""
    return (
        "### Instruction:\n"
        "You are a medical chatbot with expert knowledge in diabetes. "
        "Respond directly to the patient's question with medically accurate information. "
        "Your answer must be clear, empathetic, and professional. "
        "Do not include greetings, small talk, or compliments. "
        "Your response must be at max of 1-2 liner and no more than that.\n\n"
        f"### Patient's Question:\n{question}\n\n"
        "### Answer:"
    )

def build_prompt_guardrail(question: str, session_id: str) -> str:
    """Build prompt with conversation context matching llm/main.py"""
    history_blocks = []
    for pq, pa in _get_history_blocks(session_id):
        history_blocks.append(
            f"### Previous Question:\n{pq.strip()}\n"
            f"### Previous Answer:\n{pa.strip()}\n"
        )
    history = "".join(history_blocks)

    # Add context awareness instruction when history exists
    context_instruction = ""
    if history:
        context_instruction = "The patient is following up on the previous conversation. Answer their specific NEW question directly. DO NOT repeat your previous answer.\n\n"
        history += "\n"

    # Match the PROMPT_HEADER from llm/main.py
    return (
        "### Instruction:\n"
        "You are an expert clinical assistant focused on diabetes and closely related metabolic care. "
        "Respond with evidence-based, empathetic guidance that aligns with current standards of practice. "
        "Use no more than 40 words unless additional detail is essential for patient safety. "
        "Choose the clearest structure for the user—bulleted list, numbered steps, compact markdown table, or short paragraph—based on the question intent. "
        "Avoid greetings, avoid chit-chat, and reference prior conversation context when it improves clarity. "
        f"If the patient's question is unrelated to diabetes or human healthcare, respond exactly with '{REFUSAL_LINE}'.\n\n"
        f"{context_instruction}"
        f"{history}"
        f"### Patient's Question:\n{question}\n\n"
        "### Answer:"
    )

@app.post("/chat_stream")
def chat_stream():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or data.get("prompt") or "").strip()
    mode = (data.get("mode") or "only").lower()
    session_id = _sid()

    if not question:
        return Response("data: Error: empty prompt\n\n", mimetype="text/event-stream")

    def sse():
        t0 = time.time()
        time.sleep(0.01)
        yield f"data: __RETRIEVAL__:{(time.time()-t0):.3f}\n\n"

        # PRE-GENERATION SECURITY CHECKS (matching llm/main.py:503-598)
        if mode == "guardrail":
            # Get previous conversation context for this session
            prev_history = _get_conversation_history(session_id)
            prev_q = prev_history[-1][0] if prev_history else None
            prev_a = prev_history[-1][1] if prev_history else None
            prev_passed = True if prev_history else False

            # 1. Prompt Injection Detection
            conv_history = CONV_MEMORY.get(session_id, deque())
            recent_questions = [q for q, _, _ in conv_history]
            injection_result = detect_prompt_injection(question, recent_questions=recent_questions)

            if not injection_result.passed:
                if injection_result.message:
                    yield f"data: __WARNING__: {injection_result.message}\\n\n"
                yield f"data: {REFUSAL_LINE}\\n\n"
                yield "data: __STATUS__:done\n\n"
                _remember(session_id, question, REFUSAL_LINE, False)
                return

            # 2. Self-Harm Crisis Detection
            crisis_result = detect_self_harm_crisis(question)
            if not crisis_result.passed:
                crisis_message = crisis_response()
                if crisis_result.message:
                    yield f"data: __WARNING__: {crisis_result.message}\\n\n"
                yield f"data: {crisis_message}\\n\n"
                yield "data: __STATUS__:done\n\n"
                _remember(session_id, question, crisis_message, False)
                return

            # 3. High-Risk Medical Request Detection
            high_risk_result = detect_high_risk_medical_request(question)
            if not high_risk_result.passed:
                if high_risk_result.message:
                    yield f"data: __WARNING__: {high_risk_result.message}\\n\n"
                yield f"data: {REFUSAL_LINE}\\n\n"
                yield "data: __STATUS__:done\n\n"
                _remember(session_id, question, REFUSAL_LINE, False)
                return

            # 4. Off-Topic Question Detection (CRITICAL: use prev_q/prev_a from history!)
            is_follow_up = medical_guard._looks_like_follow_up(question, prev_q, prev_a)
            if medical_guard._is_off_topic_question(question) and not is_follow_up:
                yield f"data: {REFUSAL_LINE}\\n\n"
                yield "data: __STATUS__:done\n\n"
                _remember(session_id, question, REFUSAL_LINE, False)
                return

            # Check cache after security checks pass
            cached = _get_cached(session_id, question)
            if cached:
                yield "data: " + cached.replace("\n", "\\n") + "\n\n"
                yield "data: __STATUS__:done\n\n"
                # Cache hit still updates conversation memory!
                _remember(session_id, question, cached, True)
                return

        # Build prompt based on mode
        if mode == "guardrail":
            prompt = build_prompt_guardrail(question, session_id=session_id)
            # Balance between consistency (temp=0.0) and variety (temp=0.3+)
            # temp=0.0 from llm/main.py ensures medical accuracy but may repeat
            temperature = 0.0  # Match llm/main.py for safety
            top_p = 0.5  # Match llm/main.py
        else:
            prompt = build_prompt_model_only(question)
            temperature = 0.6
            top_p = 0.9

        t_gen0 = time.time()
        sent_ttft = False

        # GENERATION PHASE
        if mode == "guardrail":
            # Stream tokens to user while collecting for validation
            output_text = ""
            for out in llm(prompt, max_tokens=MAX_TOKENS, temperature=temperature,
                          top_p=top_p, stop=["###", "User:"], stream=True):
                token = out["choices"][0]["text"]

                # Report true TTFT on first token
                if not sent_ttft:
                    yield f"data: __TTFT__:{(time.time()-t_gen0):.3f}\n\n"
                    sent_ttft = True

                output_text += token
                # Stream tokens to user in real-time!
                yield "data: " + token.replace("\n", "\\n") + "\n\n"

            raw_text = output_text.strip()

            # Get previous context again for validation
            prev_history = _get_conversation_history(session_id)
            prev_q = prev_history[-1][0] if prev_history else None
            prev_a = prev_history[-1][1] if prev_history else None
            prev_passed = True if prev_history else False

            # POST-GENERATION VALIDATION (after streaming completes)
            passed, final_text, reason = enforce_medical_policy(
                raw_text, question, validator=medical_guard,
                previous_question=prev_q, previous_response=prev_a,
                previous_passed=prev_passed
            )

            # Apply safety filters on the validated response
            if passed:
                safety_result = enforce_safety_filters(final_text)
                if not safety_result.passed:
                    if safety_result.message:
                        yield f"data: __WARNING__: Safety filter violation: {safety_result.message}\\n\n"
                    passed = False
                    final_text = REFUSAL_LINE

            # If validation failed, send warning (user already saw the invalid response stream)
            if not passed and reason:
                yield f"data: __WARNING__: Guardrails violation: {reason}\\n\n"
                # Optionally send the refusal line
                yield f"data: [BLOCKED - {reason}]\\n\n"

            _remember(session_id, question, final_text, passed)
            if passed:
                _put_cached(session_id, question, final_text)

        else:
            # Model-only mode (no guardrails)
            output_text = ""
            for out in llm(prompt, max_tokens=MAX_TOKENS, temperature=temperature,
                          top_p=top_p, stop=["###", "User:"], stream=True):
                tok = out["choices"][0]["text"]
                if not sent_ttft:
                    yield f"data: __TTFT__:{(time.time()-t_gen0):.3f}\n\n"
                    sent_ttft = True
                output_text += tok
                yield "data: " + tok.replace("\n", "\\n") + "\n\n"

        yield "data: __STATUS__:done\n\n"

    return Response(sse(), mimetype="text/event-stream")

@app.post("/api/clear_memory")
def clear_memory():
    sid = _sid()
    CONV_MEMORY.pop(sid, None)
    # Clear all cache entries for this session
    stale = [k for k in QA_CACHE.keys() if k.startswith(f"{sid}||")]
    for k in stale:
        QA_CACHE.pop(k, None)
    return jsonify({"ok": True})

if __name__ == "__main__":
    print("Starting server on http://0.0.0.0:8001")
    app.run(host="0.0.0.0", port=8001, threaded=True)
