'''
Created on 28 Jan 2026

@author: ante
'''

import logging.config
import sys, yaml

from qdrant_client import QdrantClient
from qdrant_client.models import SearchRequest
from qdrant_client.models import Filter, FieldCondition, MatchValue

import requests

log = logging.getLogger(__name__)

with open("logging.yaml") as f:
    config = yaml.safe_load(f)
logging.config.dictConfig(config)

# Qdrant
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "springchat_tools_bg3_m3"

OLLAMA_URL = "http://localhost:11434"
MODEL = "bge-m3"   # or "bg3_m3" if that is your local alias

#
def qdrant_search(query_embedding):
    # Connect to Qdrant
    client = QdrantClient(
        url=QDRANT_URL   # or host="localhost", port=6333
    )
    
    # Search
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=5,
        with_payload=True,   # return stored metadata
        with_vectors=False
    )
    
    # Print results
    for p in results.points:
        log.info("Score: %.3f  Tool: %-20.20s  Description: %s", p.score, p.payload.get("toolName"), p.payload.get("toolDescription").replace("\n", ""))

#
def qdrant_filter_search(client: QdrantClient, query_embedding: list[float]):
    results = client.search(
    collection_name=COLLECTION_NAME,
    query_vector=query_embedding,
    limit=5,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="sessionId",
                match=MatchValue(value="session_123")
            )
        ]
    ),
    with_payload=True
)

#
def get_query_embedding(query_text: str) -> list[float]:
    response = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={
            "model": MODEL,
            "prompt": query_text
        },
        timeout=60
    )
    response.raise_for_status()
    return response.json()["embedding"]

def query_search(query_text: str):
    log.info("")
    log.info("query_text: %s", query_text)
    log.info("")
    
    query_embedding = get_query_embedding(query_text)
    #log.debug("query_text embedding: %s", len(query_embedding))
    qdrant_search(query_embedding)
    log.info("")
    log.debug("")


def main(argv: list[str] | None = None) -> int: # IGNORE:C0111
    log.info("XXXXXXXXXXXXXXXX")
    log.info(" spring-chat-py")
    
    query_search("What time is now?")
    query_search("What day is today?")    
    query_search("What have been my latest activities?")        
    query_search("Can you load whoop activities from last week?")  
    query_search("Can you make a list of all certificates?")
    
    query_search("Welche Zeit haben wir?")
    query_search("Welche Tag ist heute?")    
    query_search("Was waren die letzen Aktivitäten?")        
    query_search("Kannst du die Aktivitäten von letzer Woche von Whoop laden?")  
    query_search("Kannst du die Zertifikate auflisten?")
    
    
if __name__ == "__main__":
    sys.exit(main())