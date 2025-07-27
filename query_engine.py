from parser import parse_query
from embedder import embed_text
import chromadb
import numpy as np
import json

# Step 1: Parse user query
query = "46-year-old male, knee surgery in Pune, 3-month-old insurance policy"
parsed = parse_query(query)
print("Parsed Query:", parsed)

# Step 2: Embed query
query_embedding = embed_text(query)

# Step 3: Connect to Chroma and perform search
chroma_client = chromadb.PersistentClient(path="chroma_storage")
collection = chroma_client.get_collection("insurance_clauses")

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    include=["documents", "metadatas", "distances"]
)

# Step 4: Simple rule-based logic (mocked)
decision = "rejected"
amount = "₹0"
justification = "No matching clause found."
matched_clauses = []

for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
    text = doc.lower()
    if "knee" in text and "covered" in text and parsed['policy_duration_months'] >= 3:
        decision = "approved"
        amount = "₹1,50,000"
        justification = f"{meta['clause_id']}: {doc.strip()}"
        matched_clauses = [meta['clause_id']]
        break

# Step 5: Structured JSON
output = {
    "decision": decision,
    "amount": amount,
    "justification": justification,
    "matched_clauses": matched_clauses
}

print(json.dumps(output, indent=2))
