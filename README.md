# 🤖 LLM Experiments using Hugging Face Transformers

This repository documents my learning journey with **Large Language Models (LLMs)** through the **Hugging Face Transformers** library.  
All experiments were created and run inside **GitHub Codespaces** using Python.

---

## 🧭 Project Overview

This repo contains beginner-friendly implementations of foundational LLM tasks:
- **Zero-Shot Classification** — classify unseen text into candidate labels without training  
- **Masked Word Prediction (Fill‑Mask)** — predict missing words using BERT/RoBERTa  
- **Named Entity Recognition (NER)** — identify people, organizations, and locations in text  
- **Text Generation** — generate continuations with causal language models (e.g., GPT‑2)

Each task demonstrates how to use the `transformers` pipeline, interpret results, **and fix common errors** I actually hit while learning.

---

## 📂 Folder Structure

```
LLM_Transformer/
│
├── src/
│   ├── zero_shot.py                 # Zero-shot text classification
│   ├── mask_filling.py              # Masked word prediction
│   ├── ner_recognition.py           # Named Entity Recognition
│   ├── text_generation.py           # Text generation with GPT-2
│   └── results/                     # Saved results and output files
│       ├── zero_shot_result.txt
│       ├── mask_filling_result.txt
│       ├── ner_result.txt
│       └── text_generation_result.txt
│
├── .gitignore
└── README.md
```

> Results are saved under `src/results/` so the code and its outputs live together.

---

## 🧠 Projects Completed

### 1️⃣ Zero‑Shot Classification
**File:** `src/zero_shot.py`

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

result = classifier(
    "The Germany's weather is so unpredictable",
    candidate_labels=["education", "climate", "confused", "culture"],
)

print(result)
```
**Output (example)**
```
"labels": ["climate", "confused", "culture", "education"]
"scores": [0.87, 0.08, 0.03, 0.01]
```

**Common error & fix**
- `ModuleNotFoundError: No module named 'transformers'` → `pip install transformers torch` inside the **active venv**.

Saved to: `src/results/zero_shot_result.txt`

---

### 2️⃣ Masked Word Prediction (Fill‑Mask)
**File:** `src/mask_filling.py`

```python
from transformers import pipeline
unmasker = pipeline("fill-mask", model="bert-base-cased")   # uses [MASK]

result = unmasker("In Germany, Oktoberfest happens in this [MASK].", top_k=3)
print(result)
```
**Output (example)**
```
In Germany, Oktoberfest happens in this month.  → 0.865
In Germany, Oktoberfest happens in this city.   → 0.734
In Germany, Oktoberfest happens in this festival. → 0.622
```

**Common errors & fixes**
- `PipelineException: No mask_token ([MASK]) found on the input`  
  → I used `<mask>` by mistake; **BERT expects `[MASK]`**. (RoBERTa would use `<mask>`.)  
- `OSError: No space left on device` (Codespaces cache full)  
  → `rm -rf ~/.cache/huggingface` then rerun to redownload only what’s needed.  
- Trailing comma created a tuple accidentally: `result = unmasker(...),`  
  → remove the comma: `result = unmasker(...)`.

Saved to: `src/results/mask_filling_result.txt`

---

### 3️⃣ Named Entity Recognition (NER)
**File:** `src/ner_recognition.py`

```python
from transformers import pipeline
ner = pipeline(
    "ner",
    model="dbmdz/bert-large-cased-finetuned-conll03-english",
    aggregation_strategy="simple"   # replaces deprecated grouped_entities=True
)
print(ner("I am Shamini studying masters at FAU in Erlangen."))
```
**Output (example)**
```
Shamini → PER (Person)
FAU → ORG (Organization)
Erlangen → LOC (Location)
```

**Notes**
- Warning about `grouped_entities` being deprecated → fixed by using `aggregation_strategy="simple"`.
- The “Some weights … not used (pooler)” message is **normal** for token classification.

Saved to: `src/results/ner_result.txt`

---

### 4️⃣ Text Generation  ✅ *(added)*
**File:** `src/text_generation.py`

```python
from transformers import pipeline

# Use a causal LM for generation (not BART-MNLI which is for classification)
generator = pipeline("text-generation", model="gpt2")

prompt = "In Germany, Oktoberfest is"
outputs = generator(prompt, max_length=40, num_return_sequences=1, do_sample=True)
print(outputs[0]["generated_text"])
```

**Common pitfalls I fixed**
- ❌ Using a non‑generation model (e.g., `facebook/bart-large-mnli`) with `"text-generation"` →  
  **Fix:** Use a causal LM like `"gpt2"` or `"EleutherAI/gpt-neo-125M"` for generation tasks.
- ❌ Accidental trailing comma: `generator = pipeline(...),` → tuple, then `TypeError` when calling → **remove comma**.
- ❌ “No module named torch” → `pip install torch` in the **venv**.
- ℹ️ First run is slow while the model downloads; later runs use the cache.

Saved to: `src/results/text_generation_result.txt`

---

## 💡 How I Learned & Fixed Errors (Cheat‑Sheet)

| Challenge | What I Did |
|---|---|
| Virtual env not active | `source .venv/bin/activate` before installing/running |
| Missing packages | `pip install transformers torch` |
| Wrong mask token | `[MASK]` for BERT, `<mask>` for RoBERTa |
| “Fetch first” on `git push` | `git pull --rebase origin main` then `git push` |
| Disk full in Codespaces | `rm -rf ~/.cache/huggingface` |
| Trailing comma turned value into tuple | Remove trailing commas in assignments/calls |
| Deprecation: `grouped_entities` | Use `aggregation_strategy="simple"` |

---

## ⚙️ Install & Run

```bash
git clone https://github.com/shaminimadhanraj02-hue/LLM_Transformer.git
cd LLM_Transformer

python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1

pip install transformers torch

# Run any task
python src/zero_shot.py
python src/mask_filling.py
python src/ner_recognition.py
python src/text_generation.py
```

Results are written to `src/results/` (text/JSON), so code and outputs live together.

---

## 💾 Git Workflow I Follow

```bash
git add src/
git commit -m "Add/Update LLM experiment + saved results"
git push origin main

# If push is rejected (remote ahead):
git pull --rebase origin main
git push origin main
```

---

## 🌟 Learnings
- Core pipelines in `transformers` and when to choose each model family.  
- Matching the **right task** to the **right model** (classification vs generation).  
- Practical debugging in Codespaces (venv, caches, git rebase).  

---

## 🚀 Next
- Summarization & Question‑Answering pipelines  
- A tiny RAG demo over personal notes  
- Optional: connect a simple LLM service to ROS 2

---

## 🧾 Credits
- [Hugging Face Transformers Docs](https://huggingface.co/docs/transformers)
- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course)

---

## 👩‍💻 Author
**Shamini Madhanraj** — Master’s student (Germany) exploring AI, Robotics & LLMs.
