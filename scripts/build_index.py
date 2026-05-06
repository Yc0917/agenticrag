"""构建 FAISS + BM25 索引 + chunk store"""
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, INDEX_DIR


def build_all(
    corpus_path: str = None,
    index_dir: str = None,
    batch_size: int = 8,
    device: str = None,
):
    """从 corpus.json 构建所有索引

    Args:
        corpus_path: corpus.json 路径，默认 DATA_DIR/corpus.json
        index_dir: 索引输出目录，默认 INDEX_DIR
        batch_size: BGE-M3 编码批大小。低内存容器建议 1-8。
        device: 编码设备，默认由 embedder 使用 cpu。
    """
    if corpus_path is None:
        corpus_path = os.path.join(DATA_DIR, "corpus.json")
    if index_dir is None:
        index_dir = INDEX_DIR

    os.makedirs(index_dir, exist_ok=True)

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    print(f"[build_index] Building indexes for {len(corpus)} docs...")

    texts = [doc["text"] for doc in corpus]
    chunk_ids = [doc["chunk_id"] for doc in corpus]
    chunk_store = {doc["chunk_id"]: doc for doc in corpus}

    # 1. FAISS IndexFlatIP
    print("[build_index] Encoding with BGE-M3...")
    from retrieval.embedder import encode
    import faiss

    embeddings = encode(texts, batch_size=batch_size, device=device)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, os.path.join(index_dir, "faiss.index"))
    print(f"[build_index] FAISS index: {index.ntotal} vectors, dim={dim}")

    # 2. BM25
    print("[build_index] Building BM25 (jieba + whitespace tokenizer)...")
    from rank_bm25 import BM25Okapi
    from retrieval.keyword_search import tokenize
    tokenized = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    with open(os.path.join(index_dir, "bm25.pkl"), "wb") as f:
        pickle.dump(bm25, f)

    # 3. chunk_ids（与 FAISS 对齐）
    with open(os.path.join(index_dir, "chunk_ids.json"), "w") as f:
        json.dump(chunk_ids, f)

    # 4. chunk_store
    with open(os.path.join(index_dir, "chunk_store.pkl"), "wb") as f:
        pickle.dump(chunk_store, f)

    print(f"[build_index] All indexes saved to {index_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build FAISS + BM25 indexes")
    parser.add_argument("--corpus", default=None, help="Path to corpus.json")
    parser.add_argument("--index-dir", default=None, help="Output index directory")
    parser.add_argument("--batch-size", type=int, default=8, help="BGE-M3 encoding batch size")
    parser.add_argument("--device", default=None, help="Encoding device, e.g. cpu or cuda")
    args = parser.parse_args()
    build_all(
        corpus_path=args.corpus,
        index_dir=args.index_dir,
        batch_size=args.batch_size,
        device=args.device,
    )
