'''
Created on 1 Feb 2026

@author: ante
'''

import requests, logging
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

log = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/embed"

client = QdrantClient(url="http://localhost:6333")  # or Qdrant Cloud URL + api_key
 

def search(collection_name: str, query_text: str, model: str = "bge-m3"):
    
    payload = {"model": model, "input": query_text}
        #log.debug("payload: %s", payload)
        #log.debug("")
        # Embeddings API: input can be a list of strings
    resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
    resp.raise_for_status()
    log.debug("response: %s", resp)
    query_emeddings = resp.json()["embeddings"]
    #log.debug("body: %s", resp.json())
    log.debug(" %s", query_emeddings)
    
    #query_vector = [0.1, 0.2, 0.3, ...]  # your embedding (same dim as collection vectors)
    
    hits = client.search(
        collection_name=collection_name,
        query_vector=query_emeddings[0],
        limit=5,
        with_payload=True,
    )
    
    for h in hits:
        log.info("****************************************************************")
        log.info("")
        log.info("id: %s score: %s", h.id, h.score)
        log.info("")
        # payload often contains your text + metadata
        log.info("section: %s", h.payload.get("section"))
        log.info("text: %s", h.payload.get("text"))
