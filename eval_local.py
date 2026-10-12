# eval_local.py - compare "base model + retrieved context" vs "base + your trained adapter" on the held-out test questions
# needs in the same folder: test.jsonl (from your Drive folder WizardOS_Training) and the folder wizard_os_lora_1.7b
import os, re, json
import torch, pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen3-1.7B"
ADAPTER = "wizard_os_lora_1.7b"
N_PER_MODE = 3                      # test examples per mode (15 in total). Raise to 12 for the full set (slower).
MAX_TOK = {"Faculty: generate MCQs": 350, "Faculty: generate viva questions": 300,
           "Faculty: generate descriptive questions": 550, "Faculty: generate lab exercise": 750,
           "Explain a concept": 450}
SYSTEM = (
    "You are Wizard OS, a tutor for an undergraduate Operating Systems course. "
    "Answer using the COURSE CONTEXT provided. Prefer the terminology and algorithm versions used in the context. "
    "If the context does not cover the question, say so clearly, then give a short general answer marked as 'Outside course notes'. "
    "Never invent facts. Cite sources like [source, location]."
)

assert torch.cuda.is_available(), "No CUDA GPU found"
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb, device_map={"": 0}, dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

def generate(user_msg, max_new_tokens):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.8)
    ans = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return re.sub(r"<think>.*?</think>", "", ans, flags=re.S).strip()

test = [json.loads(l) for l in open("test.jsonl", encoding="utf-8")]
picked, seen = [], {}
for e in test:
    if seen.get(e["mode"], 0) < N_PER_MODE:
        picked.append(e); seen[e["mode"]] = seen.get(e["mode"], 0) + 1
print(len(picked), "test examples")

def fmt_ok(mode, t):
    if mode == "Faculty: generate MCQs": return bool(re.search(r"\bA[.)]", t) and re.search(r"correct answer", t, re.I))
    if mode == "Faculty: generate viva questions": return "Model answer" in t
    if mode == "Faculty: generate descriptive questions": return "Model answer outline" in t
    if mode == "Faculty: generate lab exercise": return ("Reference solution" in t) and ("```" in t)
    return None
cites = lambda t: bool(re.search(r"\[[^\]]+,\s*[^\]]+\]", t))

rows = []
for i, e in enumerate(picked, 1):
    mt = MAX_TOK[e["mode"]]
    with model.disable_adapter():
        base = generate(e["prompt"], mt)          # original model + retrieved context
    tuned = generate(e["prompt"], mt)             # with your adapter
    rows.append({"Mode": e["mode"], "Topic": e["topic"][:120], "Reference answer": e["response"],
                 "Base + RAG": base, "With adapter": tuned,
                 "Format OK (base)": fmt_ok(e["mode"], base), "Format OK (adapter)": fmt_ok(e["mode"], tuned),
                 "Cites sources (base)": cites(base), "Cites sources (adapter)": cites(tuned),
                 "Length (base)": len(base), "Length (adapter)": len(tuned)})
    print(f"{i}/{len(picked)} done: {e['mode']}", flush=True)

df = pd.DataFrame(rows)
df.to_excel("comparison.xlsx", index=False)
summary = df.groupby("Mode")[["Format OK (base)", "Format OK (adapter)", "Cites sources (base)", "Cites sources (adapter)"]].mean().round(2)
print("\nShare of answers (0 to 1):\n", summary.to_string())
print("\nSaved comparison.xlsx in this folder")
