# Wizard OS

A local, retrieval-augmented tutor for Operating Systems education. It answers questions, debugs student code, and generates MCQs, viva questions, descriptive questions and lab exercises from the course material, with sources listed under every answer.

## How it works

Course PDFs -> chunks (1,000 characters, 150 overlap) -> BGE embeddings (BAAI/bge-large-en-v1.5) -> ChromaDB -> top-k retrieval by mode -> Qwen3 (4-bit) -> answer plus sources.

A QLoRA fine-tune of Qwen3-1.7B was also tried. It improved the layout of viva and descriptive questions but made answers less accurate and less grounded, so the final app uses retrieval only. Details are in the report.

## Files

| File | Purpose |
| --- | --- |
| `Wizard_OS_Colab.ipynb` | Builds the knowledge base (parse, chunk, embed, store) and the training data |
| `app.py` | Local Streamlit app (Qwen3-4B, 4-bit) |
| `train_local.py` | QLoRA training of Qwen3-1.7B |
| `eval_local.py` | Compares base + RAG with the adapter on held-out examples |
| `Wizard_OS_Training_Review.xlsx` | The 480 drafted answers (MCQ, viva, descriptive, lab) |
| `Wizard_OS_Lab_Solutions.zip` | Tested lab solutions |
| `Wizard_OS_Project_Report.docx` | Project report |

The answers in the review sheet were drafted with Claude and reviewed by the authors. They are drafts, not faculty-verified.

## Run the app

Needs a Windows or Linux PC with an NVIDIA GPU (6 GB VRAM or more) and Python 3.10+.

1. Install PyTorch with CUDA first (https://pytorch.org), then `pip install -r requirements.txt`.
2. Build the knowledge base by running Steps 1 to 4 and 9 of `Wizard_OS_Colab.ipynb` in Colab with your course PDFs in a Drive folder. Download the saved `chroma_db` folder and put it next to `app.py`.
3. Start the app:

```
python -m streamlit run app.py --server.fileWatcherType none
```

The first run downloads Qwen3-4B (about 8 GB) and the embedding model (about 1.3 GB).

## Known limits

- Generated questions can be wrong (for example an MCQ whose answer key is incorrect). Faculty should review them.
- The evaluation used 3 held-out examples per mode and checks layout and citation presence, not correctness.
- The course PDFs and the vector database are not included in this repository.
