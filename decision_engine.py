# decision_engine.py

import torch
from embedder import embed_text
from sentence_transformers import util

def make_decision(parsed_query, retrieved_clauses):
    age = parsed_query.get("age", "")
    gender = parsed_query.get("gender", "")
    procedure = parsed_query.get("procedure", "").lower()
    location = parsed_query.get("location", "")
    duration = parsed_query.get("policy_duration_months", 0)

    if not procedure:
        return {
            "decision": "rejected",
            "amount": "₹0",
            "justification": "No procedure specified in query.",
            "matched_clauses": []
        }

    query_text = f"{age}-year-old {gender} undergoing {procedure} in {location} with {duration}-month-old policy"

    try:
        procedure_embedding = embed_text(query_text)
    except Exception as e:
        return {
            "decision": "rejected",
            "amount": "₹0",
            "justification": f"Embedding failed for procedure: {e}",
            "matched_clauses": []
        }

    best_match = None
    best_score = 0.0
    threshold = 0.4  # LOWERED threshold

    for clause in retrieved_clauses:
        text = clause.get("text", "")
        meta = clause.get("metadata", {})

        try:
            clause_embedding = embed_text(text)
        except Exception:
            continue  # Skip clause if embedding fails

        score = util.cos_sim(
            torch.tensor(procedure_embedding),
            torch.tensor(clause_embedding)
        ).item()

        print(f"[DEBUG] Clause {meta.get('clause_id', 'N/A')} Score: {score:.4f}")

        if score > best_score and score > threshold:
            best_match = (clause, score)
            best_score = score

    if best_match:
        clause, _ = best_match
        meta = clause["metadata"]
        waiting = meta.get("waiting_period_months", 0)

        if duration >= waiting:
            return {
                "decision": "approved",
                "amount": meta.get("coverage_amount", "₹1,00,000"),
                "justification": f"Matched clause {meta.get('clause_id', 'N/A')} covering the procedure.",
                "matched_clauses": [meta.get("clause_id", "N/A")]
            }

    return {
        "decision": "rejected",
        "amount": "₹0",
        "justification": f"No clause matched above similarity threshold ({threshold}).",
        "matched_clauses": []
    }
