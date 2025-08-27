from typing import List, Dict
try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

class Reranker:
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.model = CrossEncoder(model_name) if CrossEncoder else None

    def rerank(self, query: str, docs: List[Dict], top_k: int = 6) -> List[Dict]:
        if not docs: return []
        if not self.model:
            return sorted(docs, key=lambda d: d.get('score', 0.0), reverse=True)[:top_k]
        # Prepare better text for cross-encoder
        pairs = []
        for d in docs:
            text = d.get('text', '')
            if not text:
                text = d.get('chunk_summary', '')
            if not text:
                text = str(d.get('metadata', {}))
            
            # Clean and truncate text for better cross-encoder performance
            text = text[:512] if text else ''  # Limit text length
            pairs.append([query, text])
        
        scores = self.model.predict(pairs)
        for i, s in enumerate(scores):
            # Convert to positive scale and normalize
            normalized_score = max(0.0, float(s) + 10) / 10  # Shift from [-10,0] to [0,1]
            docs[i]['rerank_score'] = normalized_score
            docs[i]['score'] = normalized_score  # Also set 'score' for compatibility
        
        # If all scores are very low (cross-encoder struggled with Hebrew), 
        # use enhanced fallback scoring
        if all(d.get('rerank_score', 0) < 0.1 for d in docs):
            print("⚠️  Cross-encoder scores too low, using enhanced Hebrew-aware fallback")
            for doc in docs:
                # Enhanced scoring that considers Hebrew text quality
                text = doc.get('text', '')
                if text:
                    # Base score from text length
                    base_score = min(0.6, len(text) / 1000)
                    
                    # Boost for Hebrew text (since it's likely relevant in Hebrew documents)
                    hebrew_chars = sum(1 for c in text if c in 'אבגדהוזסעפצקרשת')
                    hebrew_boost = min(0.2, hebrew_chars / 100)  # Boost up to 0.2 for Hebrew content
                    
                    # Boost for numbers (financial data)
                    number_chars = sum(1 for c in text if c.isdigit())
                    number_boost = min(0.1, number_chars / 50)  # Boost up to 0.1 for numeric content
                    
                    # Final enhanced score
                    enhanced_score = min(0.9, base_score + hebrew_boost + number_boost)
                    doc['rerank_score'] = enhanced_score
                    doc['score'] = enhanced_score
                    
                    # Log the enhancement for debugging
                    if hebrew_boost > 0 or number_boost > 0:
                        print(f"   📊 Enhanced score: base={base_score:.3f}, Hebrew={hebrew_boost:.3f}, Numbers={number_boost:.3f} → {enhanced_score:.3f}")
        
        return sorted(docs, key=lambda d: d.get('rerank_score', 0.0), reverse=True)[:top_k]