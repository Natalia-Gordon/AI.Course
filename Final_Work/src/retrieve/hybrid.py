from typing import List, Dict, Any, Optional
from .rerank import Reranker

class HybridRetriever:
    def __init__(self, dense, sparse, reranker: Optional[Reranker] = None, namespace: str = None):
        self.dense = dense
        self.sparse = sparse
        self.reranker = reranker or Reranker()
        self.namespace = namespace

    def search(self, query: str, k_dense: int = 10, k_sparse: int = 10, metadata_filter: Dict[str, Any] = None, final_k: int = 6):
        dense_hits = []
        try:
            if self.dense:
                dense_hits = self.dense.dense_search(query, k=k_dense, namespace=self.namespace, filter=metadata_filter)
        except Exception:
            dense_hits = []
        sparse_hits = self.sparse.search(query, k=k_sparse) if self.sparse else []
        merged = {}
        for h in dense_hits + sparse_hits:
            hid = h.get('id') or h.get('chunk_id') or h.get('_id')
            if hid not in merged or h.get('score',0) > merged[hid].get('score',0):
                merged[hid] = h
        docs = list(merged.values())
        return self.reranker.rerank(query, docs, top_k=final_k)