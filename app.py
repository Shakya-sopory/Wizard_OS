"""Wizard OS - local Streamlit app (RAG + Qwen3, as in the project report).

Run:  streamlit run app.py
Needs the ChromaDB folder (saved by Step 9 of the Colab notebook) next to this file,
in a folder called  chroma_db
"""
import os, re, time
import streamlit as st
import torch
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ---------------------------------------------------------------- settings
HERE = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(HERE, "chroma_db")
COLLECTION = "wizard_os"
EMB_MODEL = "BAAI/bge-large-en-v1.5"
MODEL_NAME = "Qwen/Qwen3-4B"        # fits a 6 GB GPU in 4-bit; the report's 8B/14B need a 12 GB card
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ---------------------------------------------------------------- tutor logic (same as the notebook, Step 7)
SYSTEM = (
    "You are Wizard OS, a tutor for an undergraduate Operating Systems course. "
    "Answer using the COURSE CONTEXT provided. Prefer the terminology and algorithm versions used in the context. "
    "If the context does not cover the question, say so clearly, then give a short general answer marked as 'Outside course notes'. "
    "Never invent facts. Cite sources with the real file name and page from the context, for example [Deadlocks.pdf, p.3]."
)

MODES = {
    "Explain a concept": "Explain the concept step by step, with a small worked example where useful.",
    "Debug my code": "The student has pasted code and/or an error. Identify the specific error, explain why it happens, show the corrected code, and give one tip to avoid it. Do not just rewrite everything silently.",
    "Faculty: generate MCQs": "Write exactly {count} multiple-choice questions numbered {first} to {last}. Each has 4 options (A-D), the correct answer marked, and a one-line explanation. Exactly one option must be correct, so check every option against the context before marking it, and never ask a NOT/EXCEPT question unless exactly three options are true. Match the style of the past question papers in the context. Vary the topics and difficulty.",
    "Faculty: generate lab exercise": "Create exactly {count} different curriculum-aligned lab exercises numbered {first} to {last}. Each has: aim, problem statement, sample input/output, hints, and a reference solution.",
    "Faculty: generate viva questions": "Write exactly {count} viva questions numbered {first} to {last}, each with a short model answer. Order them from easy to hard.",
    "Faculty: generate descriptive questions": "Write exactly {count} descriptive (long-answer) exam questions numbered {first} to {last}. Each question must require a detailed explanation, for example 'Explain with a diagram...', 'Compare...', 'Solve and show each step...' or 'Discuss...'. Do not write one-line or factual-recall questions. After each question give the marks (use 10 unless the input says otherwise) and a model answer outline listing the key points and steps a full-mark answer must include. Match the style of the past question papers in the context. Vary the topics and difficulty.",
}
# mode -> (approx tokens per item, items per batch)
COUNT_MODES = {
    "Faculty: generate MCQs": (130, 10),
    "Faculty: generate lab exercise": (550, 2),
    "Faculty: generate viva questions": (110, 10),
    "Faculty: generate descriptive questions": (300, 4),
}
STYLE_CAT = {"Faculty: generate MCQs": "mcq", "Faculty: generate lab exercise": "lab",
             "Faculty: generate viva questions": "viva", "Faculty: generate descriptive questions": "descriptive"}
LEVELS = {
    "Beginner": "Use simple language, analogies, and avoid jargon. Define every term.",
    "Intermediate": "Assume the student knows basic OS terms. Be precise and concise.",
    "Advanced": "Be rigorous. Include complexity, edge cases, and trade-offs.",
}


# ---------------------------------------------------------------- loading (cached, runs once)
@st.cache_resource(show_spinner="Loading the knowledge base...")
def load_db():
    emb = SentenceTransformer(EMB_MODEL, device="cpu")     # CPU: keeps all VRAM for the LLM
    col = chromadb.PersistentClient(path=CHROMA_PATH).get_collection(COLLECTION)
    return emb, col


@st.cache_resource(show_spinner="Loading Qwen3 in 4-bit (first time downloads ~8 GB)...")
def load_llm():
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb, device_map={"": 0})
    model.eval()
    return tok, model


# ---------------------------------------------------------------- retrieval
def retrieve(emb, col, query, k=5, cats=None):
    qv = emb.encode([QUERY_PREFIX + query], normalize_embeddings=True)
    kwargs = {"where": {"cat": {"$in": cats}}} if cats else {}
    r = col.query(query_embeddings=qv.tolist(), n_results=k, **kwargs)
    return [{"text": d, "source": m["source"], "loc": m["loc"], "cat": m.get("cat"), "score": round(1 - dist, 3)}
            for d, m, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])]


def retrieve_for_mode(emb, col, question, mode, k=5):
    if mode == "Debug my code":
        hits = retrieve(emb, col, question, k, cats=["theory", "lab"])
    elif mode in STYLE_CAT:
        hits = retrieve(emb, col, question, 3, cats=[STYLE_CAT[mode]]) + retrieve(emb, col, question, k, cats=["theory"])
    else:
        hits = retrieve(emb, col, question, k, cats=["theory"])
    return hits or retrieve(emb, col, question, k)


def build_prompt(question, mode, level, hits, count=None, first=1, last=1, prior=""):
    ctx = "\n\n".join(f"[{h['source']}, {h['loc']}]\n{h['text']}" for h in hits)
    task = MODES[mode].format(count=count, first=first, last=last) if mode in COUNT_MODES else MODES[mode]
    extra = f"\n\nALREADY WRITTEN (do not repeat these):\n{prior[-3000:]}" if prior else ""
    return (f"COURSE CONTEXT:\n{ctx}\n\nTASK: {task}\nDIFFICULTY: {LEVELS[level]}{extra}\n\n"
            f"TOPIC / INPUT:\n{question}")


def generate(tok, model, user_msg, max_new_tokens):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.8)
    ans = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return re.sub(r"<think>.*?</think>", "", ans, flags=re.S).strip()


def sources_markdown(hits):
    """Sources are added by code from what was actually retrieved, not by the model."""
    seen, by_file = set(), {}
    for h in hits:
        key = (h["source"], h["loc"])
        if key in seen:
            continue
        seen.add(key)
        by_file.setdefault(h["source"], []).append(h["loc"])
    return "\n".join(f"- **{f}**: {', '.join(locs)}" for f, locs in by_file.items())


def ask(tok, model, emb, col, question, mode, level, n, progress=None):
    if mode not in COUNT_MODES:
        hits = retrieve_for_mode(emb, col, question, mode, 5)
        return generate(tok, model, build_prompt(question, mode, level, hits), 800), hits
    per_item, batch = COUNT_MODES[mode]
    n = max(1, min(int(n), 50))
    hits = retrieve_for_mode(emb, col, question, mode, min(8, 5 + n // 5))
    parts, done = [], 0
    while done < n:
        m = min(batch, n - done)
        prompt = build_prompt(question, mode, level, hits, count=m, first=done + 1, last=done + m,
                              prior="\n\n".join(parts))
        parts.append(generate(tok, model, prompt, 300 + m * per_item))
        done += m
        if progress:
            progress.progress(done / n, text=f"Generated {done} of {n}")
    return "\n\n".join(parts), hits


# ---------------------------------------------------------------- UI
st.set_page_config(page_title="Wizard OS", page_icon="🧙", layout="wide")
st.title("Wizard OS")
st.caption("Curriculum-grounded Operating Systems tutor. Runs fully on this computer.")

if not torch.cuda.is_available():
    st.error("No GPU found by PyTorch. Check that the CUDA build of torch is installed.")
    st.stop()
if not os.path.isdir(CHROMA_PATH):
    st.error(f"ChromaDB folder not found: {CHROMA_PATH}")
    st.stop()

with st.sidebar:
    mode = st.selectbox("Mode", list(MODES))
    level = st.selectbox("Difficulty", list(LEVELS), index=1)
    count = 5
    if mode in COUNT_MODES:
        count = st.number_input("How many", min_value=1, max_value=50, value=5, step=1)
    st.markdown("---")
    st.caption(f"Model: {MODEL_NAME} (4-bit)\n\nEmbeddings: {EMB_MODEL}")

placeholder = ("Paste your code and the error message here" if mode == "Debug my code" else
               "Type the topic, e.g. Round Robin scheduling" if mode in COUNT_MODES else
               "Ask an Operating Systems question")
question = st.text_area("Your question, topic, or code", height=170, placeholder=placeholder)

if st.button("Ask Wizard OS", type="primary"):
    if not question.strip():
        st.warning("Please type something first.")
    else:
        emb, col = load_db()
        tok, model = load_llm()
        bar = st.progress(0.0, text="Working...") if mode in COUNT_MODES else None
        t0 = time.time()
        with st.spinner("Thinking..."):
            answer, hits = ask(tok, model, emb, col, question, mode, level, count, bar)
        if bar:
            bar.empty()
        st.session_state["last"] = (answer, hits, time.time() - t0)

if "last" in st.session_state:
    answer, hits, secs = st.session_state["last"]
    st.markdown(answer)
    st.markdown("**Sources used from your course material:**")
    st.markdown(sources_markdown(hits))
    st.caption(f"Generated in {secs:.0f} s")
    st.download_button("Download answer (.md)", answer, file_name="wizard_os_answer.md")
