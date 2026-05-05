"""
LAB 19 - Step 3: Build Flat RAG with ChromaDB
Chunks Wikipedia corpus and indexes into ChromaDB for baseline comparison.
"""
import os
import json
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

load_dotenv()

COLLECTION_NAME = "lab19_flat_rag"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks by words."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def build_flat_rag():
    corpus_dir = Path("data/corpus")
    articles = sorted(corpus_dir.glob("*.json"))
    print(f"📚 Building Flat RAG from {len(articles)} articles...")
    
    # Init ChromaDB (persistent)
    chroma_client = chromadb.PersistentClient(path="data/chromadb")
    
    # Delete existing collection if exists
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print("🗑️  Cleared existing ChromaDB collection")
    except:
        pass
    
    embed_fn = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )
    
    all_chunks = []
    all_ids = []
    all_metas = []
    
    for article_path in articles:
        with open(article_path, encoding="utf-8") as f:
            article = json.load(f)
        
        title = article["title"]
        content = article.get("content", "")
        summary = article.get("summary", "")
        full_text = f"Title: {title}\n\n{summary}\n\n{content}"
        
        chunks = chunk_text(full_text)
        for j, chunk in enumerate(chunks):
            chunk_id = f"{article_path.stem}_{j}"
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metas.append({
                "title": title,
                "chunk_index": j,
                "url": article.get("url", ""),
            })
    
    print(f"📦 Indexing {len(all_chunks)} chunks...")
    
    # Batch insert (ChromaDB batch limit ~5000)
    BATCH = 500
    for i in range(0, len(all_chunks), BATCH):
        collection.add(
            documents=all_chunks[i:i+BATCH],
            ids=all_ids[i:i+BATCH],
            metadatas=all_metas[i:i+BATCH],
        )
        print(f"  ✅ Indexed batch {i//BATCH + 1}/{(len(all_chunks)-1)//BATCH + 1}")
    
    print(f"\n✅ Flat RAG built!")
    print(f"   🗂️  Collection: {COLLECTION_NAME}")
    print(f"   📄 Total chunks: {len(all_chunks)}")
    return collection


def query_flat_rag(question: str, n_results: int = 5) -> list[dict]:
    """Query the ChromaDB collection and return top-k chunks."""
    chroma_client = chromadb.PersistentClient(path="data/chromadb")
    embed_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = chroma_client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn
    )
    
    results = collection.query(
        query_texts=[question],
        n_results=n_results,
    )
    
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "title": meta["title"],
            "score": 1 - dist,  # cosine similarity
        })
    return chunks


if __name__ == "__main__":
    build_flat_rag()
