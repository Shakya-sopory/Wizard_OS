# train_local.py - QLoRA fine-tuning of Wizard OS on a local NVIDIA GPU (6 GB class, e.g. RTX 3060 laptop)
# usage:  python train_local.py            (put train.jsonl in the same folder)
import os, sys, json, glob, math
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                          TrainingArguments, Trainer, DataCollatorForSeq2Seq)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_NAME = "Qwen/Qwen3-1.7B"      # use "Qwen/Qwen3-1.7B" if you run out of memory
MAXLEN = 1536                     # examples longer than this are dropped; raise to 1536 if memory allows
EPOCHS = 1
DATA = "train.jsonl"
OUT = "wizard_os_lora_" + MODEL_NAME.split("-")[-1].lower()

SYSTEM = (
    "You are Wizard OS, a tutor for an undergraduate Operating Systems course. "
    "Answer using the COURSE CONTEXT provided. Prefer the terminology and algorithm versions used in the context. "
    "If the context does not cover the question, say so clearly, then give a short general answer marked as 'Outside course notes'. "
    "Never invent facts. Cite sources like [source, location]."
)

assert torch.cuda.is_available(), "No CUDA GPU found. Install the CUDA build of PyTorch."
print("GPU:", torch.cuda.get_device_name(0), "| VRAM: %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1e9))

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb, device_map={"": 0},
                                             torch_dtype=torch.bfloat16)

rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]
def encode(ex):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ex["prompt"]}]
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    p_ids = tok(p, add_special_tokens=False)["input_ids"]
    r_ids = tok(ex["response"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    ids = p_ids + r_ids
    if len(ids) > MAXLEN:
        return None
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": [-100] * len(p_ids) + r_ids}
data = [e for e in (encode(r) for r in rows) if e]
print(f"Kept {len(data)} of {len(rows)} examples (<= {MAXLEN} tokens)")
if len(data) < 300:
    print("WARNING: fewer than 300 examples kept; consider raising MAXLEN or shortening the prompts.")

class DS(torch.utils.data.Dataset):
    def __len__(self): return len(data)
    def __getitem__(self, i): return data[i]

model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()

total_steps = math.ceil(len(data) / 8) * EPOCHS
args = TrainingArguments(
    output_dir="checkpoints", per_device_train_batch_size=1, gradient_accumulation_steps=8,
    num_train_epochs=EPOCHS, learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=max(1, int(0.05 * total_steps)),
    bf16=True, logging_steps=5, save_steps=5, save_total_limit=2, optim="adamw_torch",
    report_to="none", remove_unused_columns=False, dataloader_num_workers=0)
trainer = Trainer(model=model, args=args, train_dataset=DS(),
                  data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
resume = bool(glob.glob("checkpoints/checkpoint-*"))      # rerun the script to continue after an interruption
trainer.train(resume_from_checkpoint=resume or None)
model.save_pretrained(OUT); tok.save_pretrained(OUT)
print("TRAINING COMPLETE. Adapter saved in the folder:", OUT)
