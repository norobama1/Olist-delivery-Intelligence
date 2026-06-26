""" Joins Olist review text to delivery outcomes, embeds review
comments with a multilingual model, and caches the result for retrieval
 """

import pickle
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer

REVIEWS_PATH = Path("data/olist_order_reviews_dataset.csv")
PROCESSED_PATH = Path("data/olist_processed.csv")
CACHE_PATH = Path("rag/embeddings_cache.pkl")
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

def load_labeled_reviews() -> pd.DataFrame:
    reviews = pd.read_csv(REVIEWS_PATH, usecols=['order_id','review_score','review_comment_message'])
    reviews = reviews.dropna(subset=['review_comment_message'])
    reviews['review_comment_message'] = reviews['review_comment_message'].astype(str).str.lower().str.strip()
    reviews = reviews[reviews['review_comment_message'].str.len() > 15]
    reviews = reviews.drop_duplicates(subset="order_id", keep="first")

    delays = pd.read_csv(PROCESSED_PATH, usecols=['order_id','delayed'])
    merged = reviews.merge(delays,on='order_id',how='inner')
    return merged.reset_index(drop=True)

def build_and_cache() -> None:
    df = load_labeled_reviews()
    print(f"Embedding {len(df)} labeled reviews ({df['delayed'].mean():.1f} from delayed orders)") 

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        df["review_comment_message"].tolist(),
        show_progress_bar = True,
        batch_size = 64,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    CACHE_PATH.parent.mkdir(parents=True,exist_ok=True) 
    with open(CACHE_PATH,"wb") as f:
        pickle.dump({"df":df,"embeddings": embeddings,"model_name": MODEL_NAME}, f)

    print(f"Cached to {CACHE_PATH}")

if __name__ == "__main__":
    build_and_cache()