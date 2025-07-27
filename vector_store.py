# vector_store.py

import os
import re
import glob
import chromadb
from chromadb.config import Settings
from PyPDF2 import PdfReader
from embedder import embed_text

# Setup
PDF_FOLDER = os.path.join(os.getcwd(), "documents")

if os.path.exists("chroma_storage"):
    import shutil
    shutil.rmtree("chroma_storage")

chroma_client = chromadb.PersistentClient(path="chroma_storage")
collection = chroma_client.create_collection(name="insurance_clauses")

# Clause-like info extractors
def extract_waiting_period(text):
    match = re.search(r"(\d+)\s*month[s]?\s*(waiting|wait)?", text, re.IGNORECASE)
    return int(match.group(1)) if match else 0

def extract_coverage_amount(text):
    match = re.search(r"₹\s?[\d,]+", text)
    return match.group(0) if match else "₹1,00,000"

def chunk_text_into_clauses(raw_text):
    # Prefer "Clause X" splits, else fallback to paragraphs
    clause_chunks = re.split(r"(Clause\s*\d+[\.\d]*[:.\-]?)", raw_text, flags=re.IGNORECASE)

    if len(clause_chunks) > 1:
        # Combine every clause label with its body
        clauses = []
        for i in range(1, len(clause_chunks) - 1, 2):
            title = clause_chunks[i].strip()
            body = clause_chunks[i + 1].strip()
            if len(body) > 40:
                clauses.append(f"{title} {body}")
        return clauses

    # Fallback: split by paragraphs or 2+ newlines
    paras = [p.strip() for p in re.split(r'\n\s*\n', raw_text) if len(p.strip()) > 50]
    return paras

# Load PDFs
pdf_paths = glob.glob(os.path.join(PDF_FOLDER, "*.pdf"))
print(f"🔍 Found {len(pdf_paths)} PDFs in {PDF_FOLDER}\n")

doc_id = 1
total_indexed = 0

for path in pdf_paths:
    print(f"📄 Processing: {os.path.basename(path)}")
    reader = PdfReader(path)
    full_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    print("📜 Sample Extract:\n", full_text[:500], "\n")

    clauses = chunk_text_into_clauses(full_text)
    print(f"✅ Found {len(clauses)} usable chunks\n")

    for clause_text in clauses:
        waiting = extract_waiting_period(clause_text)
        amount = extract_coverage_amount(clause_text)

        embedding = embed_text(clause_text)
        collection.add(
            documents=[clause_text],
            embeddings=[embedding],
            metadatas=[{
                "clause_id": f"AUTO-{doc_id}",
                "waiting_period_months": waiting,
                "coverage_amount": amount
            }],
            ids=[f"clause-{doc_id}"]
        )

        doc_id += 1
        total_indexed += 1

print(f"✅ Done! Indexed {total_indexed} chunks from {len(pdf_paths)} PDFs.")
