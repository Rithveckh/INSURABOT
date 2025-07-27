import os
import pdfplumber

def extract_text_chunks_from_pdf(pdf_path, chunk_size=500):
    all_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n"

    paragraphs = all_text.split("\n\n")
    chunks = []
    buffer = ""

    for para in paragraphs:
        sentences = para.strip().split(". ")
        for sentence in sentences:
            buffer += sentence.strip() + ". "
            if len(buffer) >= chunk_size:
                chunks.append(buffer.strip())
                buffer = ""
        if buffer:
            chunks.append(buffer.strip())
            buffer = ""

    return chunks

def load_all_documents(folder_path="documents"):
    all_chunks = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            full_path = os.path.join(folder_path, filename)
            print(f"📄 Processing: {filename}")
            chunks = extract_text_chunks_from_pdf(full_path)
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "source": filename,
                    "chunk_id": f"{filename}-{i}",
                    "text": chunk
                })
    return all_chunks

# Test
if __name__ == "__main__":
    results = load_all_documents()
    print(f"✅ Total Chunks Extracted: {len(results)}")
    print("🧩 Sample:", results[0])
