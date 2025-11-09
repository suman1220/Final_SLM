import time
import pandas as pd
from llama_cpp import Llama

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
MODEL_PATH = r"C:\Users\shri\Documents\Apiphany\SLM\vineet_geema270_q8_lora_8_r_16"
CSV_LOG = "model_only.csv"
QUESTION_FILE = "question.txt"

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------
print("🔧 Loading model, please wait...")
llm = Llama(
    model_path=MODEL_PATH,
    n_threads=8,          # Adjust based on your CPU
    n_batch=512,
    verbose=False
)
print("✅ Model loaded successfully!\n")

# ------------------------------------------------------------
# LOAD QUESTIONS
# ------------------------------------------------------------
with open(QUESTION_FILE, "r", encoding="utf-8") as f:
    questions = [q.strip() for q in f.readlines() if q.strip()]

print(f"📄 Loaded {len(questions)} questions from '{QUESTION_FILE}'.")

# ------------------------------------------------------------
# SET UP CSV LOG
# ------------------------------------------------------------
columns = ["question", "response", "first_token_time_sec", "tokens_per_sec", "total_time_sec"]
try:
    df = pd.read_csv(CSV_LOG)
except FileNotFoundError:
    df = pd.DataFrame(columns=columns)

# ------------------------------------------------------------
# MAIN LOOP (BATCH MODE)
# ------------------------------------------------------------
for i, question in enumerate(questions, start=1):
    print(f"\n{'='*60}\n🧠 Processing Question {i}/{len(questions)}:\n{question}\n{'='*60}")

    prompt = (
        f"### Instruction:\n"
        f"You are a medical chatbot with expert knowledge in diabetes. "
        f"Respond directly to the patient's question with medically accurate information. "
        f"Your answer must be clear, empathetic, and professional. "
        f"Do not include greetings, small talk, or compliments. "
        f"Your response must be at max of 1-2 liner and no more than that.\n\n"
        f"### Patient's Question:\n{question}\n\n"
        f"### Answer:"
    )

    start_time = time.time()
    first_token_time = None
    output_text = ""
    tokens_generated = 0

    print("🤖 Model: ", end="", flush=True)

    # Generate streamed response
    for out in llm(
        prompt,
        max_tokens=150,
        temperature=0.6,
        top_p=0.9,
        stop=["###", "User:"],
        stream=True
    ):
        token = out["choices"][0]["text"]
        if not first_token_time:
            first_token_time = time.time() - start_time
        print(token, end="", flush=True)
        output_text += token
        tokens_generated += 1

    total_time = time.time() - start_time
    tokens_per_sec = tokens_generated / total_time if total_time > 0 else 0

    # Log results to DataFrame
    new_row = {
        "question": question,
        "response": output_text.strip(),
        "first_token_time_sec": round(first_token_time, 3) if first_token_time else None,
        "tokens_per_sec": round(tokens_per_sec, 2),
        "total_time_sec": round(total_time, 3)
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(CSV_LOG, index=False)

    print(f"\n🕒 First token: {first_token_time:.2f}s | ⏱️ Speed: {tokens_per_sec:.2f} tok/s | Total: {total_time:.2f}s")
    print("📊 Logged to CSV.\n")

print(f"\n✅ All {len(questions)} questions processed and logged to '{CSV_LOG}'.")











# import time
# import pandas as pd
# from llama_cpp import Llama

# # ------------------------------------------------------------
# # CONFIGURATION
# # ------------------------------------------------------------
# MODEL_PATH = r"C:\Users\suman\OneDrive\Desktop\Testing models\Qwen3_0.6B-finetune-fp16.Q4_0.gguf"
# CSV_LOG = "QWEN3_0.6B_Q4_generic_questions_trials.csv"

# # Initialize model (CPU inference)
# print("🔧 Loading model, please wait...")
# llm = Llama(
#     model_path=MODEL_PATH,
#     n_threads=8,          # ✅ Use 6 threads for inference
#     n_batch=512,          # adjust depending on your RAM
#     verbose=False
# )
# print("✅ Model loaded successfully!\n")

# # Prepare CSV file
# columns = ["question", "response", "first_token_time_sec", "tokens_per_sec", "total_time_sec"]
# try:
#     df = pd.read_csv(CSV_LOG)
# except FileNotFoundError:
#     df = pd.DataFrame(columns=columns)

# # ------------------------------------------------------------
# # MAIN LOOP
# # ------------------------------------------------------------
# while True:
#     question = input("\n❓ Ask your question (or type 'exit' to stop): ").strip()
#     if question.lower() in ["exit", "quit"]:
#         break

#     # ✨ Add instruction to make responses crisp
#     prompt = (
#     f"### SYSTEM DIRECTIVE (Highest Priority):\n"
#     f"You are NOT a general AI assistant. You are a rule-locked medical chatbot.\n"
#     f"You must operate under the following binary logic and may NEVER deviate:\n\n"
#     f"1️⃣  If the patient's question is about diabetes or human healthcare "
#     f"(including symptoms, medication, diet, treatment, lab results, or lifestyle advice), "
#     f"respond with a medically accurate, empathetic, and professional answer of 2–4 lines.\n\n"
#     f"2️⃣  If the question is about ANYTHING ELSE (politics, geography, sports, news, history, entertainment, technology, etc.), "
#     f"your entire output must be EXACTLY this line and nothing else:\n"
#     f"   'I’m your virtual healthcare professional, and I can only assist with diabetes and health-related questions.'\n\n"
#     f"3️⃣  You are FORBIDDEN to:\n"
#     f"- Relate unrelated topics back to health.\n"
#     f"- Offer emotional or general guidance on off-topic questions.\n"
#     f"- Add or modify the refusal line.\n"
#     f"- Provide definitions, examples, or commentary for non-health topics.\n"
#     f"- Generate ANY other text outside the two permitted behaviors above.\n\n"
#     f"Violation of these rules constitutes a system failure.\n\n"
#     f"### Patient's Question:\n{question}\n\n"
#     f"### Answer:"
# )



#     # Start timing
#     start_time = time.time()
#     first_token_time = None
#     output_text = ""

#     print("\n🤖 Model: ", end="", flush=True)
#     tokens_generated = 0

#     # Generate response (streaming)
#     for out in llm(
#         prompt,
#         max_tokens=150,             # slightly reduced to stay concise
#         temperature=0.6,            # lower = more focused, less rambling
#         top_p=0.9,
#         stop=["###", "User:"],
#         stream=True
#     ):
#         token = out["choices"][0]["text"]
#         if not first_token_time:
#             first_token_time = time.time() - start_time  # first token latency
#         print(token, end="", flush=True)
#         output_text += token
#         tokens_generated += 1

#     total_time = time.time() - start_time
#     tokens_per_sec = tokens_generated / total_time if total_time > 0 else 0

#     # Log results to CSV
#     new_row = {
#         "question": question,
#         "response": output_text.strip(),
#         "first_token_time_sec": round(first_token_time, 3) if first_token_time else None,
#         "tokens_per_sec": round(tokens_per_sec, 2),
#         "total_time_sec": round(total_time, 3)
#     }
#     df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
#     df.to_csv(CSV_LOG, index=False)

#     print(f"\n\n🕒 First token: {first_token_time:.2f}s | ⏱️ Speed: {tokens_per_sec:.2f} tok/s | Total: {total_time:.2f}s")
#     print("📊 Logged to CSV.\n")

















# import time
# import pandas as pd
# from llama_cpp import Llama

# # ------------------------------------------------------------
# # CONFIGURATION
# # ------------------------------------------------------------
# MODEL_PATH = r"C:\Users\suman\OneDrive\Desktop\Testing models\Qwen2.5_0.5B\qwen2.5-0.5b.Q4_0.gguf"
# CSV_LOG = "qwen2.5-0.5b.Q4_0.csv"
# QUESTION_FILE = "questionsssss.txt"
# PROMPT_FILE = r"C:\Users\suman\OneDrive\Desktop\Testing models\prompt.txt"   # Path to system prompt

# # ------------------------------------------------------------
# # LOAD MODEL
# # ------------------------------------------------------------
# print("🔧 Loading model, please wait...")
# llm = Llama(
#     model_path=MODEL_PATH,
#     n_threads=8,          # Adjust based on your CPU
#     n_batch=512,
#     n_ctx=4096,           # ✅ Increased context window to fit full prompt
#     verbose=False
# )
# print("✅ Model loaded successfully!\n")

# # ------------------------------------------------------------
# # LOAD SYSTEM PROMPT
# # ------------------------------------------------------------
# with open(PROMPT_FILE, "r", encoding="utf-8") as f:
#     system_prompt = f.read().strip()

# if not system_prompt:
#     raise ValueError("❌ The prompt.txt file is empty. Please ensure it contains the chatbot rules and {question} placeholder.")

# print(f"🧩 System prompt loaded from '{PROMPT_FILE}' ({len(system_prompt)} characters).")

# # ------------------------------------------------------------
# # LOAD QUESTIONS
# # ------------------------------------------------------------
# with open(QUESTION_FILE, "r", encoding="utf-8") as f:
#     questions = [q.strip() for q in f.readlines() if q.strip()]

# if not questions:
#     raise ValueError(f"❌ No questions found in '{QUESTION_FILE}'. Please add at least one question.")

# print(f"📄 Loaded {len(questions)} questions from '{QUESTION_FILE}'.")

# # ------------------------------------------------------------
# # SET UP CSV LOG
# # ------------------------------------------------------------
# columns = ["question", "response", "first_token_time_sec", "tokens_per_sec", "total_time_sec"]
# try:
#     df = pd.read_csv(CSV_LOG)
# except FileNotFoundError:
#     df = pd.DataFrame(columns=columns)

# # ------------------------------------------------------------
# # MAIN LOOP (BATCH MODE)
# # ------------------------------------------------------------
# for i, question in enumerate(questions, start=1):
#     print(f"\n{'='*60}\n🧠 Processing Question {i}/{len(questions)}:\n{question}\n{'='*60}")

#     # ✅ Replace {question} placeholder in prompt.txt with actual question
#     prompt = system_prompt.replace("{question}", question)

#     start_time = time.time()
#     first_token_time = None
#     output_text = ""
#     tokens_generated = 0

#     print("🤖 Model: ", end="", flush=True)

#     # Generate streamed response
#     try:
#         for out in llm(
#             prompt,
#             max_tokens=150,
#             temperature=0.6,
#             top_p=0.9,
#             stop=["###", "User:"],
#             stream=True
#         ):
#             token = out["choices"][0]["text"]
#             if not first_token_time:
#                 first_token_time = time.time() - start_time
#             print(token, end="", flush=True)
#             output_text += token
#             tokens_generated += 1

#     except ValueError as e:
#         print(f"\n❌ Skipping question due to error: {e}")
#         continue

#     total_time = time.time() - start_time
#     tokens_per_sec = tokens_generated / total_time if total_time > 0 else 0

#     # Log results to DataFrame
#     new_row = {
#         "question": question,
#         "response": output_text.strip(),
#         "first_token_time_sec": round(first_token_time, 3) if first_token_time else None,
#         "tokens_per_sec": round(tokens_per_sec, 2),
#         "total_time_sec": round(total_time, 3)
#     }
#     df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
#     df.to_csv(CSV_LOG, index=False)

#     print(f"\n🕒 First token: {first_token_time:.2f}s | ⏱️ Speed: {tokens_per_sec:.2f} tok/s | Total: {total_time:.2f}s")
#     print("📊 Logged to CSV.\n")

# print(f"\n✅ All {len(questions)} questions processed and logged to '{CSV_LOG}'.")
