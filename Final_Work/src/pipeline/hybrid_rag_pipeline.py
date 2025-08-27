from pathlib import Path
import yaml
from index.pinecone_index import PineconeIndexer
from index.tfidf_index import TfidfIndexer
from retrieve.rerank import Reranker
from retrieve.hybrid import HybridRetriever

class HybridRAGPipeline:
    """Simple wrapper that builds indices from chunks in data/processed and provides a .query() API."""
    def __init__(self, config_path: str = 'src/config.yaml'):
        cfg = yaml.safe_load(Path(config_path).read_text())
        self.cfg = cfg
        # load chunks if exist
        processed = Path('data/processed/ayalon_qN_2024/chunks.jsonl')
        chunks = []
        if processed.exists():
            for ln in processed.read_text(encoding='utf-8').splitlines():
                try:
                    chunks.append(__import__('json').loads(ln))
                except:
                    pass
        else:
            # fallback to data/documents/demo
            demo = Path('data/documents/report_q2_anonymized.md')
            if demo.exists():
                chunks.append({'id':'demo_0','text':demo.read_text(encoding='utf-8'),'page_number':1,'section_type':'Analysis','file_name':demo.name})
        # build indices (Pinecone init requires env vars; if not present, dense will be None)
        try:
            self.dense = PineconeIndexer(index_name=cfg['pinecone']['index_name'], dimension=cfg['embedding']['dim'], environment=cfg['pinecone']['environment'], metric=cfg['pinecone']['metric'])
            if chunks:
                self.dense.upsert_chunks(chunks, namespace=cfg['pinecone']['namespace'])
        except Exception as e:
            print('Pinecone init failed or missing API key; dense retrieval disabled. Error:', e)
            self.dense = None
        self.sparse = TfidfIndexer()
        if chunks:
            self.sparse.fit(chunks)
        self.reranker = Reranker(model_name=cfg.get('reranker',{}).get('model','cross-encoder/ms-marco-MiniLM-L-6-v2'))
        self.retriever = HybridRetriever(self.dense, self.sparse, self.reranker, namespace=cfg['pinecone']['namespace'])

    def query(self, question: str, k_dense: int = None, k_sparse: int = None):
        cfg = self.cfg.get('retrieval',{})
        k_dense = k_dense or cfg.get('dense_k',10)
        k_sparse = k_sparse or cfg.get('sparse_k',10)
        hits = self.retriever.search(question, k_dense=k_dense, k_sparse=k_sparse, final_k=cfg.get('final_k',6))
        # normalize to expected response format
        response = {
            'answer': '\n'.join([h.get('chunk_summary') or h.get('text','')[:300] for h in hits]),
            'contexts': hits
        }
        return response