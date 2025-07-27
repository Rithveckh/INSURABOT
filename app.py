import streamlit as st
from parser import parse_query
from embedder import embed_text
from decision_engine import make_decision
import chromadb
import json

# UI Config
st.set_page_config(page_title="🛡️ InsuraBot", layout="centered")

# Header
st.markdown("<h1 style='text-align:center;'>🛡️ InsuraBot</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>An LLM-Powered Insurance Claim Eligibility Checker</h4>", unsafe_allow_html=True)
st.markdown("---")

# Example Tip
st.markdown("💡 **Example:** `46-year-old male, knee surgery in Pune, 3-month-old policy`")

# Query Input
query = st.text_area("🔍 **Enter your query**", placeholder="Describe age, gender, treatment, city, and policy duration...")

if query:
    # Step 1: Parse Query
    parsed = parse_query(query)

    # Step 2: Embed
    query_embedding = embed_text(query)

    # Step 3: ChromaDB
    chroma_client = chromadb.PersistentClient(path="chroma_storage")
    collection = chroma_client.get_collection("insurance_clauses")

    # Step 4: Vector Search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )

    matching_clauses = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        matching_clauses.append({"text": doc, "metadata": meta})

    # Step 5: Decision
    output = make_decision(parsed, matching_clauses)

    # Step 6: Parsed Query Output
    st.subheader("📤 Parsed Query")
    st.json(parsed)

    # Step 7: Styled Decision Output
    if output["decision"] == "approved":
        st.success(f"✅ **Approved** – {output['justification']}")
    else:
        st.error(f"❌ **Rejected** – {output['justification']}")

    st.markdown(f"💰 **Claim Amount:** {output['amount']}")
    st.markdown(f"📄 **Matched Clause IDs:** {', '.join(output['matched_clauses']) if output['matched_clauses'] else 'None'}")

    # Step 8: Sample Human-Friendly Response
    st.subheader("🗣️ Sample Response")
    procedure = parsed.get("procedure", "your treatment").capitalize()
    location = parsed.get("location", "your city")

    if output["decision"] == "approved":
        st.success(f"✅ Yes, **{procedure}** is covered under the policy in **{location}**.")
    else:
        st.warning(f"❌ No, **{procedure}** is not covered under the policy based on current information.")

    # Step 9: Clause Viewer
    st.subheader("📄 Top Matching Clauses")
    for clause in matching_clauses:
        with st.expander(f"Clause {clause['metadata'].get('clause_id', 'N/A')}"):
            st.markdown(clause["text"].strip())

else:
    st.info("✍️ Enter a query above to check insurance eligibility.")
