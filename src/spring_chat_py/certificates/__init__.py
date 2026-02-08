import json, logging, requests, uuid
from pathlib import Path
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct

log = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).resolve().parent
JSON_PATH  = BASE_DIR / "certificates.json"
OLLAMA_URL = "http://localhost:11434/api/embed"

client = QdrantClient(url="http://localhost:6333")


def load_certificates() -> List[Dict[str, Any]]:
    log.debug("load_certificates")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)
    
    chunks: List[Dict[str, Any]] = []    
    
    for e in data:
        log.debug("%s %s", e["id"], e["Short-Name"])
        content = e["Short-Name"] + " "+ e["Name"] + " " + e["Department"] + " " + e["University"] + " " + e["Issued"]
        chunk={
            "content":  content,
            "metadata": {
                "Short-Name": e["Short-Name"],
                "Name":       e["Name"],
                "Department": e["Department"],
                "University": e["University"],
                "Issued":     e["Issued"],
            }
        }
        chunks.append(chunk)
        
    return chunks

def embed_chunks(
    chunks: List[Dict[str, Any]],
    model: str      = "bge-m3",
    batch_size: int = 64,
) -> List[List[float]]:
    """
    Returns embeddings in the same order as `chunks`.
    Also stores them into each chunk under item["embeddings"] and item["embedding_model"].
    """
    log.debug("embed_chunks: %d chunks, model=%s, batch_size=%d", len(chunks), model, batch_size)

    vectors: List[List[float]] = []

    for b in range(0, len(chunks), batch_size):
        batch = chunks[b : b + batch_size]
        inputs = [c["content"] for c in batch]

        payload = {"model": model, "input": inputs}
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()

        data = resp.json()
        embs = data["embeddings"]

        if len(embs) != len(batch):
            raise ValueError(f"Embedding count mismatch: got {len(embs)} embeddings for {len(batch)} inputs")

        # preserve order: zip(batch, embs) matches inputs order
        for item, emb in zip(batch, embs):
            item["embedding_model"] = model
            item["embeddings"] = emb
            vectors.append(emb)
            
    return vectors


def recreate_collection(collection_name: str, vector_size: int = 1024):
    log.debug("recreate_collection")
    
    recreate: bool     = True
    distance: Distance = Distance.COSINE
    
    existing = client.get_collections().collections
    exists = any(c.name == collection_name for c in existing)

    if recreate and exists:
        client.delete_collection(collection_name=collection_name)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance),
        )
    
    payloads:   List[Dict[str, Any]]  = load_certificates()
    log.info("payload len: %s", len(payloads))
    
    embeddings: List[List[float]]     = embed_chunks(payloads)
    if len(payloads) != len(embeddings):
        raise ValueError("Payload / embedding count mismatch")
    
    points: List[PointStruct] = []

    for payload, vector in zip(payloads, embeddings):
        #log.debug("metadata: %s", payload["metadata"])
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "doc_content": payload["content"],
                "metadata": payload["metadata"]
            }
        )
        points.append(point)
    
    client.upsert(collection_name=collection_name, points=points)
    log.info("certificates uploaded to Qdrant Vector Store %s", collection_name)

