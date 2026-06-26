"""
Semantic search over the cached review embedding,with an optimal filter on delivery 
outcome so you can compare language across delayed vs on-time orders
"""

import pickle
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer, util

CACHE_PATH = Path("rag/embeddings_cache.pkl")

_cache = None
_model = None

def _load():
    global _cache, _model
    if _cache is None:
        with open(CACHE_PATH,"rb") as f:
            _cache = pickle.load(f)
    _model = SentenceTransformer(_cache["model_name"])
    return _cache , _model

def search_reviews(query: str, filter_delayed: int | None=None, top_k: int=15) -> pd.DataFrame:
    """
    query: search text eg. "atraso na entrega"
    filter_delayed: if 1 filter on delayed; if 0 on on-time; if None no filter
    top_k: number of similar reviews to return
    """
    cache, model = _load()
    df, embeddings = cache['df'], cache['embeddings']

    if filter_delayed is not None:
        mask = df['delayed'] == filter_delayed
        df, embeddings = df[mask].reset_index(drop=True), embeddings[mask.values]

    query_embedding = model.encode([query],convert_to_numpy=True,normalize_embeddings=True)
    hits = util.semantic_search(query_embedding,embeddings,top_k=top_k)[0]

    results = df.iloc[[h['corpus_id'] for h in hits]].copy()
    results['similarity'] =[h['score'] for h in hits]
    return results[['order_id','review_score','review_comment_message','delayed','similarity']]

if __name__ == "__main__":
    results = search_reviews("prazo de entrega muito curto", filter_delayed=1, top_k=10)
    print(results.to_string(index=False))
    
