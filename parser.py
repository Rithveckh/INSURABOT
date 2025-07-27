import re
import spacy
import os
import json
import subprocess
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Ensure en_core_web_sm is available ---
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Groq setup (optional)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
USE_GROQ = GROQ_API_KEY is not None

if USE_GROQ:
    from groq import Groq
    groq_client = Groq(api_key=GROQ_API_KEY)
    GROQ_MODEL = "llama3-70b-8192"  # or "llama3-8b-8192"

# --- Fallback regex-based parser ---
def fallback_parser(query):
    age_match = re.search(r'(\d+)[- ]?year[- ]?old', query.lower())
    if not age_match:
        age_match = re.search(r'age[: ]?(\d+)', query.lower())
    if not age_match:
        age_match = re.search(r'(\d+)\s?(m|f|male|female)\b', query.lower())
    age = int(age_match.group(1)) if age_match else None

    gender = None
    if match := re.search(r'\b(male|female)\b', query.lower()):
        gender = match.group(1)
    elif match := re.search(r'\d+\s?(m|f)\b', query.lower()):
        g = match.group(1).lower()
        gender = "male" if g == "m" else "female"

    procedures = [
        "knee surgery", "cardiac surgery", "gallbladder removal",
        "hip replacement", "bypass", "cataract", "appendix removal",
        "root canal", "appendicitis surgery"
    ]
    procedure = next((p for p in procedures if p in query.lower()), None)

    loc_match = re.search(r"in\s+([a-zA-Z\s]+)", query.lower())
    location = loc_match.group(1).strip() if loc_match else None

    duration_match = re.search(r"(\d+)[- ]?month[s]?", query.lower())
    policy_duration = int(duration_match.group(1)) if duration_match else None

    return {
        "age": age,
        "gender": gender,
        "procedure": procedure,
        "location": location,
        "policy_duration_months": policy_duration
    }

# --- Groq or fallback parser ---
def parse_query(query):
    if USE_GROQ:
        try:
            prompt = f"""
Extract the following fields from the user's insurance query in JSON format:
- age (int or null)
- gender ("male", "female", or null)
- procedure (string or null)
- location (string or null)
- policy_duration_months (int or null)

Respond ONLY with a raw JSON object.

User Query: "{query}"
"""
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )

            content = response.choices[0].message.content.strip()
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            else:
                print("⚠️ No JSON block found in Groq output. Falling back...")
        except Exception as e:
            print("❌ Groq parsing failed:", e)

    # Use regex fallback
    return fallback_parser(query)

# --- Test ---
if __name__ == "__main__":
    q = "46M, cardiac surgery in Mumbai, 6 months"
    print(parse_query(q))
