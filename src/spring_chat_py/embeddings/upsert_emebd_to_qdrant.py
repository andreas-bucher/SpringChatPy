'''
Created on 30 Jan 2026

@author: ante
'''
from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from qdrant_client.models import PointStruct
from pathlib import Path
import logging, json

from itertools import islice
from typing import Any, Dict, Iterable, Iterator, List, Optional

#from qdrant_client import QdrantClient
#from qdrant_client.models import Distance, PointStruct, VectorParams

log = logging.getLogger(__name__)

def iter_jsonl_points(
    jsonl_dir: str,
    *,
    id_field: str = "id",
    vector_field: str = "embeddings",
    payload_field: str = "metadata",
    text_field: str = "text",
    skip_empty_vectors: bool = True,
) -> Iterator[PointStruct]:
    """
    Reads *.jsonl files in a directory and yields Qdrant PointStruct objects.

    Expected per-line JSON shape (flexible):
      {
        "id": "...",
        "embedding": [0.1, ...],
        "metadata": {...},
        "text": "optional chunk text",
        "embedding_model": "optional"
      }
    """
    for file_path in sorted(Path(jsonl_dir).glob("*.jsonl")):
        with file_path.open("r", encoding="utf-8") as f:
            log.info("file: %s", file_path)
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                data: Dict[str, Any] = json.loads(line)

                if id_field not in data:
                    raise ValueError(f"Missing '{id_field}' in {file_path}:{line_no}")

                pid = str(data[id_field])
                log.info("id_field: %s", pid)

                vec = data.get(vector_field)
                if vec is None:
                    raise ValueError(f"Missing '{vector_field}' in {file_path}:{line_no} id={pid}")

                if skip_empty_vectors and (not isinstance(vec, list) or len(vec) == 0):
                    # you can `raise` instead if you prefer hard-fail
                    continue

                payload = dict(data.get(payload_field, {}) or {})

                # Helpful for RAG: keep text in payload if present
                if text_field in data and "text" not in payload:
                    payload["text"] = data[text_field]

                # Preserve model info if present
                if "embedding_model" in data:
                    payload.setdefault("embedding_model", data["embedding_model"])

                yield PointStruct(id=pid, vector=vec, payload=payload)


def batched(it: Iterable[PointStruct], batch_size: int) -> Iterator[List[PointStruct]]:
    it = iter(it)
    while batch := list(islice(it, batch_size)):
        yield batch


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    *,
    distance: Distance = Distance.COSINE,
    recreate: bool = False,
) -> None:
    """
    Ensures collection exists with given vector size. Use recreate=True if you want to wipe & rebuild.
    """
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


def upload_jsonl_dir(
    client: QdrantClient,
    collection_name: str,
    jsonl_dir: str,
    *,
    batch_size: int = 256,
) -> None:
    """
    Uploads JSONL directory into Qdrant using batched upserts.
    """
    point_iter = iter_jsonl_points(jsonl_dir)

    # Peek first point to infer vector size and ensure collection exists
    first_point: Optional[PointStruct] = None
    for first_point in point_iter:
        break

    if first_point is None:
        raise RuntimeError(f"No valid points found in directory: {jsonl_dir}")

    vector_size = len(first_point.vector)
    ensure_collection(client, collection_name, vector_size, distance=Distance.COSINE, recreate=False)

    # Upload first + rest in batches
    def chain_first() -> Iterator[PointStruct]:
        yield first_point  # type: ignore[misc]
        yield from point_iter

    for points in batched(chain_first(), batch_size):
        client.upsert(collection_name=collection_name, points=points)


def upload(collection_name: str, in_dir: str):
    client = QdrantClient(url="http://localhost:6333")

    upload_jsonl_dir(
        client=client,
        collection_name=collection_name,
        jsonl_dir=in_dir,
        batch_size=256
    )

    log.info("✅ Upload done")