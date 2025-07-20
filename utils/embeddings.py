from typing import List, Union, Dict
import numpy as np
import os
from pinecone import Pinecone, ServerlessSpec
import uuid

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_chunk(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()

    def embed_chunks(
        self,
        chunks: Union[List[str], List[Dict]],
        normalize: bool = False,
        return_with_text: bool = False
    ) -> Union[List[List[float]], List[Dict]]:
        results = []
        for chunk in chunks:
            text = chunk["text"] if isinstance(chunk, dict) else chunk
            embedding = self.embed_chunk(text)

            if normalize:
                norm = np.linalg.norm(embedding)
                embedding = (np.array(embedding) / norm).tolist()

            if return_with_text:
                results.append({
                    "text": text,
                    "embedding": embedding
                })
            else:
                results.append(embedding)

        return results
    

class PineconeVectorStore:
    def __init__(self, api_key: str = None, index_name: str = "rag-index", dimension: int = 384):
        api_key = api_key or os.getenv("PINECONE_API_KEY")
        pc = Pinecone(api_key=api_key)

        if not pc.has_index(index_name):
            pc.create_index(
                name=index_name,
                vector_type="dense",
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                ),
                deletion_protection="disabled",
                tags={
                    "environment": "development"
                }
            )
        self.index = pc.Index(index_name)

    def upsert_embeddings(self, items: List[Dict], document: str, repo: str) -> None:
        """
        items = [
            {
                'text': '...',
                'embedding': [...],
                'metadata': { opcional }
            },
            ...
        ]
        """
        vectors = []
        for i, item in enumerate(items):
            vec_id = f"chunk-{uuid.uuid4().hex[:8]}"
            if isinstance(item, dict):
                text = item.get("text", "")
                embedding = item.get("embedding", [])

                vectors.append({
                    "id": vec_id,
                    "values": embedding,
                    "metadata": {"text": text, "chunk_index": i, "document":document, "repo": repo}
                })
            else:
                vectors.append({
                    "id": vec_id,
                    "values": item,
                    "metadata": {"text": "", "chunk_index": i, "document":document, "repo": repo}
                })

        self.index.upsert(vectors=vectors)
        print(f"[Pinecone] Insertados {len(vectors)} vectores.")

    def query(self, embedding: List[float], top_k: int = 5) -> List[Dict]:
        results = self.index.query(vector=embedding, top_k=top_k, include_metadata=True)
        return results.get("matches", [])