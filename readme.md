# Applied Large Language Models in the Social Sciences

A **one-day applied workshop** introducing social scientists to the practical use of Large Language Models (LLMs) in research. Participants gain hands-on experience across three distinct applied projects, each demonstrating a different mode of working with LLMs — from training a small model from scratch, to querying a frontier model via API, to analysing images with a multimodal model.

The workshop is aimed at **advanced masters and PhD students** in the social sciences. No deep technical background is assumed, but basic familiarity with Python and data analysis is helpful.

---

## Schedule

| Time | Session |
|------|---------|
| 10:00–10:50 | Part 1: Introduction to LLMs |
| 10:50–11:00 | *Break* |
| 11:00–12:00 | Project 1: Training & Validating a Small Model |
| 12:00–13:00 | *Lunch* |
| 13:00–14:00 | Project 2: Structured Data Extraction with a Large Model API |
| 14:00–14:10 | *Break* |
| 14:10–14:50 | Project 3: Analysing Images with a Vision Model |
| 14:50–15:00 | Wrap-up & Discussion |

---

## Pre-requisites

Please complete all of the steps below **before arriving at the workshop**. Each step takes 5–10 minutes. If you run into any problems, email the instructor in advance.

---

### 1. Google Account & Google Colab

All notebooks run on **Google Colab** — a free, browser-based Python environment with access to GPU computing. You do not need to install anything locally.

**Steps:**

1. Make sure you have a **Google account**. If you don't have one, create one at [accounts.google.com](https://accounts.google.com).
2. Go to [colab.research.google.com](https://colab.research.google.com) and sign in.
3. Click **New notebook** to create a blank notebook.
4. In the first cell, type `print("Hello, world!")` and press **Shift + Enter** to run it.
5. You should see `Hello, world!` appear below the cell. If so, you are all set.

> **GPU access:** For Project 1 (model training) you will need a GPU runtime. In Colab, go to **Runtime → Change runtime type → T4 GPU** and click Save. The free tier includes sufficient GPU time for this workshop.

---

### 2. OpenAI API Key

Projects 2 and 3 use the **OpenAI API** (GPT-4o). You will need your own API key and a small amount of credit loaded onto your account. The total cost of running both projects is approximately **$0.10–$0.50**.

**Steps:**

1. Go to [platform.openai.com](https://platform.openai.com) and create an account (or sign in if you already have one).
2. Navigate to **Settings → Billing** and add a payment method. Load a minimum of **$5 credit** — this is more than enough for the workshop.
3. Navigate to **API Keys** (in the left-hand menu or at [platform.openai.com/api-keys](https://platform.openai.com/api-keys)).
4. Click **Create new secret key**, give it a name (e.g. `workshop`), and click **Create**.
5. **Copy the key immediately** — it starts with `sk-` and will only be shown once. Paste it somewhere safe (e.g. a password manager or a private notes file).

> **Keep your key private.** Do not share it, post it in a chat, or paste it into a notebook you plan to share. If you accidentally expose a key, delete it immediately at the API Keys page and create a new one.

**Test your key (optional but recommended):**

Paste the following into a Colab cell, replacing `sk-...` with your actual key, and run it:

```python
from openai import OpenAI
client = OpenAI(api_key="sk-...")   # replace with your key
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say: API key works."}],
    max_tokens=10,
)
print(response.choices[0].message.content)
```

You should see `API key works.` If you see an authentication error, double-check that you copied the full key and that billing is set up.

---

### 3. Course Materials

The notebooks and datasets are hosted in this repository.

**Steps:**

1. Download the repository as a ZIP file (click **Code → Download ZIP** on the GitHub page) and unzip it, **or** clone it with:
   ```bash
   git clone https://github.com/YOUR-REPO-URL.git
   ```
2. Upload the three notebook files and the `data/` folder to your Google Drive or directly to Colab when prompted during the workshop.

Alternatively, each notebook contains instructions for loading files directly within Colab — the instructor will walk through this at the start of each project.

---

### Summary Checklist

| Task | Done? |
|------|-------|
| Google account created | ☐ |
| Google Colab opens and runs a test cell | ☐ |
| GPU runtime enabled in Colab | ☐ |
| OpenAI account created | ☐ |
| Billing set up with ≥ $5 credit | ☐ |
| API key created and saved securely | ☐ |
| API key tested successfully | ☐ |
| Course materials downloaded | ☐ |

---

## Part 1 — Introduction to Large Language Models (50 min)

**Objectives:** Establish a conceptual foundation for the rest of the workshop. No coding — slides and discussion only.

### Topics

- **What is a language model?**
  Intuition via autocomplete. Predicting the next token from context. Why this is surprisingly powerful.

- **From word embeddings to transformers**
  Word2Vec and early embeddings. The limitations of bag-of-words. The attention mechanism and why it changed everything. BERT vs. GPT: encoder vs. decoder models.

- **The landscape of modern LLMs**
  Pre-trained foundation models (BERT, GPT, Llama). Frontier models (ChatGPT, Claude, Gemini). Open vs. closed models and what this means for research.

- **Applications in the social sciences**
  Text classification, topic modelling, content analysis, information extraction, coding at scale. Survey of recent published examples (political science, sociology, economics).

- **Key research considerations**
  Validity and reliability of LLM-based measurement. Bias and representational harms. Cost, reproducibility, and transparency. When to fine-tune vs. when to prompt.

---

## Project 1 — Training & Validating a Small Model (60 min)

**Research question:** Can we automatically classify news articles by topic using a fine-tuned BERT model?

**Mode:** Fine-tuning a pre-trained transformer on labelled data. This project illustrates the full supervised learning pipeline, including validation.

### Dataset
BBC News corpus (2,225 articles, five categories: Business, Entertainment, Politics, Sport, Tech).

### Steps

1. **Load and explore the data** — inspect the corpus, class balance, and a sample of articles
2. **Prepare the data** — train/test split, tokenisation with the BERT tokeniser, conversion to HuggingFace `Dataset` format
3. **Fine-tune the model** — load `bert-base-cased`, attach a classification head, train with HuggingFace `Trainer`
4. **Validate the model** — accuracy, precision, recall, F1 score, confusion matrix
5. **Run inference on new text** — classify unseen articles using the trained pipeline
6. **Discuss** — What makes a good classifier? How should social scientists report and interpret these results?

### Tools & Libraries
- `transformers` (HuggingFace)
- `datasets`
- `scikit-learn`
- `PyTorch`

### Notebook
`code/bbc_news_cats.ipynb`

### Key learning outcomes
- Understand the fine-tuning paradigm and how it differs from training from scratch
- Run a full training and evaluation pipeline
- Interpret standard classification metrics in a social science context
- Appreciate the relationship between labelled data quality and model performance

---

## Project 2 — Structured Data Extraction with a Large Model API (60 min)

**Research question:** Can we use a frontier LLM to extract structured information from unstructured text at scale?

**Mode:** Prompting a large model via API (no training required). This project contrasts with Project 1 — rather than fine-tuning, we use prompt engineering to direct a pre-trained model.

### Dataset
UK House of Commons Private Members' Bills (`data/hoc_private_member_bills.csv`). Each row is a parliamentary bill with a short title and description. The task is to extract structured fields: policy area, key actors, and legislative intent.

### Steps

1. **Introduction to the OpenAI API** — authentication, `client.chat.completions.create()`, understanding system vs. user messages
2. **Prompt engineering basics** — zero-shot vs. few-shot prompting, instructing the model to return structured output (JSON)
3. **Design the extraction schema** — define the fields to extract and write a clear prompt
4. **Run extraction over the corpus** — loop through records, call the API, parse responses
5. **Validate the output** — compare a sample of API extractions against manual coding; compute agreement
6. **Discuss** — reliability and consistency across runs, handling failures, cost at scale, and when this approach is preferable to fine-tuning

### Tools & Libraries
- `openai` (Python SDK)
- `pandas`
- `json`

### Notebook
`code/api_extraction.ipynb` *(to be created)*

### Key learning outcomes
- Set up and use the OpenAI API from Python
- Write effective prompts for information extraction tasks
- Return and parse structured (JSON) outputs from LLMs
- Evaluate LLM-based extraction against a human-coded gold standard
- Understand cost, rate limits, and reproducibility when using commercial APIs

---

## Project 3 — Analysing Images with a Vision Model (40 min)

**Research question:** Can we use a multimodal LLM to systematically code the content of political images?

**Mode:** Querying a vision-capable model (e.g. GPT-4o) via the OpenAI API with both text prompts and images. This project extends the API skills from Project 2 to a multimodal setting.

### Dataset
A curated set of political and social images (e.g. protest photographs, political campaign imagery, newspaper front pages). *(Dataset to be assembled — candidate sources include Wikimedia Commons and publicly available political image archives.)*

### Steps

1. **What can vision models see?** — brief overview of multimodal LLMs and how image + text inputs work
2. **Research applications** — political imagery, protest coding, visual framing analysis, archival photograph analysis, social media content moderation research
3. **Send an image to the API** — load an image, encode it, construct a multimodal prompt
4. **Structured visual coding** — prompt the model to return a coded response (e.g. presence of actors, emotional tone, setting, dominant theme)
5. **Run across a small corpus** — loop over images, collect coded outputs, aggregate into a dataframe
6. **Discuss** — limitations (hallucination, cultural bias, sensitivity), ethical considerations, comparison with human coders, reproducibility

### Tools & Libraries
- `openai` (with vision support)
- `PIL` / `base64` for image handling
- `pandas`

### Notebook
`code/vision_analysis.ipynb` *(to be created)*

### Key learning outcomes
- Understand what multimodal models can and cannot do
- Send image inputs to an LLM API alongside text prompts
- Design a structured visual coding scheme and implement it via prompting
- Critically assess the validity of automated image analysis for social science

---

## Wrap-up & Discussion (10 min)

- Where is the field heading? Agentic systems, fine-tuning vs. prompting at scale, open-source alternatives
- Validation and robustness: why measurement quality matters as much as model choice
- Responsible use: transparency, reproducibility, and reporting standards for LLM-assisted research
- Resources for further learning (HuggingFace documentation, papers, community)
- Open Q&A

---

## Learning Goals

By the end of the workshop, participants will be able to:

1. Explain how LLMs work at a conceptual level and distinguish between key model types (encoder, decoder, multimodal).
2. Fine-tune a pre-trained transformer for a supervised classification task and evaluate its performance.
3. Use a commercial LLM API to perform structured text extraction via prompt engineering.
4. Query a vision-capable model to code the content of images for research purposes.
5. Critically assess the validity, reliability, and ethical implications of LLM-based measurement in social science.

---

## Repository Contents

| Path | Description |
|------|-------------|
| `slides/main.qmd` | Quarto source for the lecture slides |
| `slides/main.html` | Compiled slide deck |
| `code/bbc_news_cats.ipynb` | Project 1: BERT fine-tuning and validation |
| `code/api_extraction.ipynb` | Project 2: Structured extraction via OpenAI API |
| `code/vision_analysis.ipynb` | Project 3: Image analysis with a vision model |
| `data/bbc-news-data.csv` | BBC News dataset (Project 1) |
| `data/hoc_private_member_bills.csv` | UK parliamentary bills dataset (Project 2) |

---

## Technical Requirements

Participants should have Python ≥ 3.9 installed. All notebooks are designed to run on **Google Colab** (no local GPU required).

### For Projects 1
```
pip install transformers[torch] datasets evaluate accelerate scikit-learn pandas matplotlib
```

### For Projects 2 & 3
```
pip install openai pandas pillow
```

An **OpenAI API key** is required for Projects 2 and 3. Participants should create an account and add a small amount of credit before the workshop. Costs for the exercises are expected to be minimal (< $1 total).

---

**Zach Dickson** — Fellow in Quantitative Methodology, London School of Economics
