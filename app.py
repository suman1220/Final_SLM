# app.py
import os, time
from typing import Optional
from flask import Flask, request, Response, jsonify, send_from_directory
from llama_cpp import Llama
from collections import deque
from typing import Deque, Tuple

# ---- Guardrail policy (used in "Model + Guardrail") ----
# Files you provided:
# - guardrails_medical.py (policy + enforce_medical_policy)
# - main_for_guardrail.py (prompting & conversation ideas we mirror)
from guardrails_medical import (
    MedicalResponsePolicy,
    enforce_medical_policy
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", r"C:\Users\shri\Documents\Apiphany\SLM\merged-Q8_0 (1).gguf")  # <-- set this!
N_THREADS  = int(os.getenv("LLM_THREADS", "6"))
N_BATCH    = int(os.getenv("LLM_BATCH", "256"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "180"))

app = Flask(__name__, static_folder=".", static_url_path="")

# -----------------------------------------------------------------------------
# Load model once (same model for all three modes)
# -----------------------------------------------------------------------------
print("🔧 Loading model…")
llm = Llama(
    model_path=MODEL_PATH,
    n_threads=N_THREADS,
    n_batch=N_BATCH,
    verbose=False
)
print("✅ Model loaded!")

# -----------------------------------------------------------------------------
# Conversation memory (minimal; used only by guardrail follow-ups)
# -----------------------------------------------------------------------------
HISTORY_WINDOW = 2
# previous_question: Optional[str] = None
# previous_response: Optional[str] = None
# previous_passed: bool = False
# conversation_history: list[tuple[str, str]] = []
medical_guard = MedicalResponsePolicy()
REFUSAL_LINE = MedicalResponsePolicy.REFUSAL_LINE

# ---- Per-session short memory & small TTL cache ----
# Memory: last N turns per session (for guardrail context)
CONV_MEMORY: dict[str, Deque[Tuple[str, str, bool]]] = {}

# Cache: (session_id, normalized_question) -> (final_answer, timestamp)
QA_CACHE: dict[tuple[str, str], tuple[str, float]] = {}
CACHE_TTL_SECONDS = 60 * 30  # 30 minutes
MEMORY_MAXLEN = 20           # keep last 20 Q/A
HISTORY_WINDOW = 2           # only pass last 2 to prompt (same as before)

def _normalize_q(q: str) -> str:
    return " ".join(q.split()).lower()

def _sid() -> str:
    # Prefer X-Session-Id from the browser; fall back to 'anon'
    return request.headers.get("X-Session-Id", "anon")

def _get_history_blocks(session_id: str) -> list[tuple[str, str]]:
    dq = CONV_MEMORY.get(session_id)
    if not dq:
        return []
    # take last HISTORY_WINDOW *passed* answers only
    hist: list[tuple[str, str]] = []
    for q, a, passed in list(dq)[-HISTORY_WINDOW:]:
        if passed and q and a:
            hist.append((q, a))
    return hist

def _remember(session_id: str, question: str, answer: str, passed: bool):
    dq = CONV_MEMORY.setdefault(session_id, deque(maxlen=MEMORY_MAXLEN))
    dq.append((question, answer, passed))

def _get_cached(session_id: str, question: str):
    key = (session_id, _normalize_q(question))
    v = QA_CACHE.get(key)
    if not v:
        return None
    ans, ts = v
    if (time.time() - ts) > CACHE_TTL_SECONDS:
        QA_CACHE.pop(key, None)
        return None
    return ans

def _put_cached(session_id: str, question: str, answer: str):
    key = (session_id, _normalize_q(question))
    QA_CACHE[key] = (answer, time.time())

# -----------------------------------------------------------------------------
# Small config for Settings drawer (kept for compatibility)
# -----------------------------------------------------------------------------
@app.get("/api/config")
def api_config():
    return jsonify({
        "model": os.path.basename(MODEL_PATH),
        "ctx": 4096, "threads": N_THREADS, "batch": N_BATCH,
        "k": 2, "chunk_size": 300, "chunk_overlap": 60, "context_budget": 480,
        "faiss_hnsw": False, "hnsw_m": 0, "hnsw_ef_search": 0
    })

# -----------------------------------------------------------------------------
# Serve the UI (your index2.html)
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    return send_from_directory(".", "index2.html")

# -----------------------------------------------------------------------------
# Helpers: prompts for the two modes you asked for
# -----------------------------------------------------------------------------
def build_prompt_model_only(question: str) -> str:
    # Mirrors your main_only_model style: short, direct answer. :contentReference[oaicite:3]{index=3}
    return (
        "### Instruction:\n"
        "You are a medical chatbot with expert knowledge in diabetes. "
        "Respond directly to the patient's question with medically accurate information. "
        "Your answer must be clear, empathetic, and professional. "
        "Do not include greetings, small talk, or compliments. "
        "Your response must be at max of 1–2 lines and no more than that.\n\n"
        f"### Patient's Question:\n{question}\n\n"
        "### Answer:"
    )

def build_prompt_guardrail(question: str, session_id: str) -> str:
    # Build short history from per-session memory
    history_blocks = []
    for pq, pa in _get_history_blocks(session_id):
        history_blocks.append(
            "### Previous Question:\n" + pq.strip() + "\n"
            "### Previous Answer:\n" + pa.strip() + "\n"
        )
    history = "".join(history_blocks)
    if history:
        history += "\n"

    return (
        "### Instruction:\n"
        "You are a medical chatbot with expert knowledge in diabetes management. "
        "Keep answers concise (2–4 sentences), avoid greetings, and reference prior context when helpful. "
        f"If the patient's question is unrelated to diabetes or human healthcare, respond exactly with: '{REFUSAL_LINE}'.\n\n"
        f"{history}"
        f"### Patient's Question:\n{question}\n\n"
        "### Answer:"
    )


# -----------------------------------------------------------------------------
# Core chat endpoint (SSE)
# -----------------------------------------------------------------------------
@app.post("/chat_stream")
def chat_stream():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or data.get("prompt") or "").strip()
    mode = (data.get("mode") or "only").lower()
    session_id = _sid()

    if not question:
        return Response("data: Error: empty prompt\n\n", mimetype="text/event-stream")

    def sse():
        # --- Retrieval timing placeholder (keep your marker) ---
        t0 = time.time()
        time.sleep(0.01)
        yield f"data: __RETRIEVAL__:{(time.time()-t0):.3f}\n\n"

        # --- Quick cache hit (guardrail only) ---
        if mode == "guardrail":
            cached = _get_cached(session_id, question)
            if cached:
                # Return cached safe text immediately
                yield "data: " + cached.replace("\n", "\\n") + "\n\n"
                yield "data: __STATUS__:done\n\n"
                return

        # --- Build the right prompt (now using per-session history for guardrail) ---
        if mode == "guardrail":
            prompt = build_prompt_guardrail(question, session_id=session_id)
            temperature = 0.3
        else:
            prompt = build_prompt_model_only(question)
            temperature = 0.6  # default & rag for now

        # --- Stream tokens from model (unchanged) ---
                # --- Generate ---
        t_gen0 = time.time()
        sent_ttft = False

        if mode == "guardrail":
            # 1) run non-stream to capture full text
            out = llm(
                prompt,
                max_tokens=MAX_TOKENS,
                temperature=temperature,
                top_p=0.9,
                stop=["###", "User:"],
                stream=False,
            )
            if not sent_ttft:
                yield f"data: __TTFT__:{(time.time()-t_gen0):.3f}\n\n"
                sent_ttft = True

            raw_text = out["choices"][0]["text"]

            # 2) guardrail
            prev = CONV_MEMORY.get(session_id)
            prev_q = prev[-1][0] if prev and len(prev) else None
            prev_a = prev[-1][1] if prev and len(prev) else None
            prev_passed = prev[-1][2] if prev and len(prev) else False

            passed, final_text, reason = enforce_medical_policy(
                raw_text,
                question,
                validator=medical_guard,
                previous_question=prev_q,
                previous_response=prev_a,
                previous_passed=prev_passed,
            )

            # 3) remember & cache
            _remember(session_id, question, final_text, passed)
            _put_cached(session_id, question, final_text)

            # 4) emit ONLY the final text once
            yield "data: " + final_text.replace("\n", "\\n") + "\n\n"

        else:
            # original streaming path for model-only / rag
            output_text = ""
            for out in llm(
                prompt,
                max_tokens=MAX_TOKENS,
                temperature=temperature,
                top_p=0.9,
                stop=["###", "User:"],
                stream=True,
            ):
                tok = out["choices"][0]["text"]
                if not sent_ttft:
                    yield f"data: __TTFT__:{(time.time()-t_gen0):.3f}\n\n"
                    sent_ttft = True
                output_text += tok
                yield "data: " + tok.replace("\n", "\\n") + "\n\n"

        yield "data: __STATUS__:done\n\n"

    return Response(sse(), mimetype="text/event-stream")

@app.post("/clear_memory")
def clear_memory():
    sid = _sid()
    CONV_MEMORY.pop(sid, None)
    # drop cache entries for just this session
    stale = [k for k in QA_CACHE.keys() if k[0] == sid]
    for k in stale:
        QA_CACHE.pop(k, None)
    return jsonify({"ok": True})

if __name__ == "__main__":
    # Run: set MODEL_PATH env var to your .gguf, then:
    #   python app.py
    app.run(host="0.0.0.0", port=8001, threaded=True)
