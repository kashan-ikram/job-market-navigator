import gradio as gr
import faiss
import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
import os

# ── Load embedding model ──────────────────────────────────────────
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ── Load FAISS index + chunks from HuggingFace repo ──────────────
print("Loading FAISS index...")
index_path  = hf_hub_download(repo_id="kashanikram/job-market-navigator-distilgpt2",
                               filename="faiss_index.bin", repo_type="model")
chunks_path = hf_hub_download(repo_id="kashanikram/job-market-navigator-distilgpt2",
                               filename="chunks.pkl", repo_type="model")

index = faiss.read_index(index_path)
with open(chunks_path, "rb") as f:
    chunks = pickle.load(f)

print(f"FAISS index loaded — {index.ntotal} vectors")

# ── Load fine-tuned model ─────────────────────────────────────────
print("Loading fine-tuned model...")
BASE_MODEL      = "distilgpt2"
FINETUNED_MODEL = "kashanikram/job-market-navigator-distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float32
)
model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL)
model.eval()
print("Model loaded!")

# ── RAG search ────────────────────────────────────────────────────
def search_rag(query, top_k=3):
    query_embedding = embedder.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

def format_rag_answer(results):
    lines = []
    for chunk in results:
        info = {}
        for line in chunk.strip().split("\n"):
            if ":" in line:
                key, _, val = line.partition(":")
                info[key.strip()] = val.strip()
        title    = info.get("Job Title", "")
        company  = info.get("Company", "")
        location = info.get("Location", "")
        salary   = info.get("Salary", "Not specified")
        desc     = info.get("Description", "")[:150]
        if title:
            lines.append(f"• {title} at {company} ({location})\n  Salary: {salary}\n  {desc}")
    return "\n\n".join(lines) if lines else "No results found."

# ── Model generation ──────────────────────────────────────────────
def generate_answer(prompt):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=350
    )
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            repetition_penalty=1.5,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### Response:" in full:
        response = full.split("### Response:")[-1]
    else:
        response = full[len(prompt):]
    response = response.split("### End")[0].strip()
    response = response.split("### Instruction")[0].strip()
    return response if response else "No answer generated."

# ── Agentic router ────────────────────────────────────────────────
def agent_router(query):
    query_lower = query.lower()
    rag_keywords   = ["salary","company","location","toronto","vancouver","canada",
                      "hiring","job opening","current","latest","remote","hybrid","available"]
    model_keywords = ["skills","requirements","how to","advice","career","learn",
                      "prepare","resume","interview","qualify","what should"]
    rag_score   = sum(1 for k in rag_keywords   if k in query_lower)
    model_score = sum(1 for k in model_keywords if k in query_lower)
    if rag_score > model_score:
        return "rag"
    elif model_score > rag_score:
        return "model"
    return "both"

def run_agent(query):
    route = agent_router(query)
    if route in ("rag", "both"):
        results = search_rag(query)
        answer  = format_rag_answer(results)
    else:
        prompt = f"### Instruction:\n{query}\n\n### Response:\n"
        answer = generate_answer(prompt)
    return route, answer

# ── Gradio UI ─────────────────────────────────────────────────────
def chat(query):
    if not query or not query.strip():
        return "Please enter a question!"
    route, answer = run_agent(query)
    label = {"rag": "Live Job Data (RAG)",
             "model": "AI Model",
             "both": "RAG + AI Model"}.get(route, route)
    return f"[Source: {label}]\n\n{answer}"

demo = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(
        placeholder="e.g. Which companies are hiring AI engineers in Canada?",
        label="Ask about the job market",
        lines=2
    ),
    outputs=gr.Textbox(label="Answer", lines=10),
    title="AI Job Market Navigator",
    description="Ask about Canadian tech jobs, salaries, companies hiring, and skills needed. Powered by RAG + Fine-tuned LLM + Agentic routing.",
    examples=[
        ["Which companies are hiring AI engineers in Canada right now?"],
        ["What is the salary for data scientist in Toronto?"],
        ["What skills do I need for machine learning engineer?"],
        ["Which companies are hiring software engineers in Vancouver?"],
    ]
)

if __name__ == "__main__":
    demo.launch()
