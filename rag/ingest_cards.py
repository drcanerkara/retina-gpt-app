import os
import json
import glob
import numpy as np
import faiss
from openai import OpenAI

DATA_DIR = "data/corpus_cards"
INDEX_PATH = "data/index.faiss"
META_PATH = "data/meta.json"

def chunk_cards(md_text: str):
    # Split by top-level headings "# "
    parts = md_text.split("\n# ")
    chunks = []
    for i, p in enumerate(parts):
        p = p.strip()
        if not p:
            continue
        if i == 0 and p.startswith("# "):
            text = p
        else:
            text = "# " + p
        chunks.append(text)
    return chunks

def embed_text(client: OpenAI, text: str):
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return resp.data[0].embedding

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY env var not set. (CMD: set OPENAI_API_KEY=sk-xxxx)")

    client = OpenAI(api_key=api_key)

    files = glob.glob(os.path.join(DATA_DIR, "*.md"))
    if not files:
        raise RuntimeError(f"No .md files found in {DATA_DIR}. Put retina_cards.md there.")

    meta = []
    embeddings = []

    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = chunk_cards(content)
        for ch in chunks:
            meta.append({"text": ch, "source_file": os.path.basename(fpath)})

    print(f"Found {len(meta)} chunks. Creating embeddings...")

    for i, item in enumerate(meta, start=1):
        emb = embed_text(client, item["text"])
        embeddings.append(emb)
        if i % 10 == 0 or i == len(meta):
            print(f"Embedded {i}/{len(meta)}")

    X = np.array(embeddings, dtype="float32")
    dim = X.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(X)

    os.makedirs("data", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("✅ Index created:")
    print(" - data/index.faiss")
    print(" - data/meta.json")

if __name__ == "__main__":
    main()
