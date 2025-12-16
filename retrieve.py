import re
import zlib
import numpy as np
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Any
from rank_bm25 import BM25Okapi
from database import JSONLDataset

class RRFProcessor:
    def __init__(self, k: int = 60):
        self.k = k
        
    def fuse(self, ranked_lists: List[List[int]]) -> List[Tuple[int, float]]:
        rrf_scores = defaultdict(float)
        for ranking in ranked_lists:
            for rank, doc_id in enumerate(ranking):
                rrf_scores[doc_id] += 1.0 / (self.k + rank + 1)
        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

class Retriever:
    def __init__(self, embedder, hybrid_embedder: bool = True):
        """
        embedder: объект с методом embed(text) -> (dense, sparse)
        hybrid_embedder: 
            True -> берем sparse от эмбеддера (BGE-M3).
            False -> считаем sparse сами через BM25 (FastText).
        """
        self.embedder = embedder
        self.hybrid_embedder = hybrid_embedder
        self.rrf = RRFProcessor()
        
        self.docs_store: Dict[int, Dict[str, Any]] = {}
        self.dense_index = []
        self.sparse_index: List[Dict[int, float]] = []
        self.ids_map: List[int] = []
        
        self.bm25_model = None
        
    def _normalize_dense(self, vector: List[float]) -> np.ndarray:
        arr = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            return arr / norm
        return arr
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Нормальная токенизация через RegEx.
        1. (?u) - поддержка Unicode (чтобы работало с кириллицей)
        2. \b - граница слова
        3. \w+ - само слово (буквы или цифры)
        """
        tokens = re.findall(r"(?u)\b\w+\b", text.lower())
        
        return tokens

    def _hash_token(self, token: str) -> int:
        return zlib.crc32(token.encode('utf-8'))

    def _build_bm25_doc_vector(self, text: str) -> Dict[int, float]:
        """Расчет разреженного вектора вручную через BM25"""
        tokens = self._tokenize(text)
        doc_len = len(tokens)
        freqs = Counter(tokens)
        
        vector = {}
        k1 = 1.5
        b = 0.75
        avgdl = self.bm25_model.avgdl
        
        for token, freq in freqs.items():
            if token in self.bm25_model.idf:
                idf = self.bm25_model.idf[token]
                numerator = freq * (k1 + 1)
                denominator = freq + k1 * (1 - b + b * doc_len / avgdl)
                weight = idf * (numerator / denominator)
                
                vector[self._hash_token(token)] = weight
                
        return vector

    def index_datasets(self, datasets_paths: Dict[str, str]):
        if not self.hybrid_embedder:
            print("Preparing BM25 corpus...")
            corpus_tokens = []
            for _, path in datasets_paths.items():
                dataset = JSONLDataset(path)
                for i in range(len(dataset)):
                    record = dataset[i]
                    text = record.get("chunk_text") or record.get('question') or ""
                    corpus_tokens.append(self._tokenize(text))
            
            print(f"Training BM25 on {len(corpus_tokens)} chunks...")
            self.bm25_model = BM25Okapi(corpus_tokens)
            del corpus_tokens

        global_idx = 0
        for source_name, path in datasets_paths.items():
            print(f"Indexing {source_name} from {path}")
            dataset = JSONLDataset(path)
            
            for i in range(len(dataset)):
                record = dataset[i]
                text_to_embed = record.get("chunk_text") or record.get('question') or ""
                
                dense_vec, sparse_vec = self.embedder.embed(text_to_embed)

                if not self.hybrid_embedder:
                    sparse_vec = self._build_bm25_doc_vector(text_to_embed)
                
                self.dense_index.append(self._normalize_dense(dense_vec))
                self.sparse_index.append(sparse_vec)
                
                self.docs_store[global_idx] = {
                    "source": source_name,
                    "content": record
                }
                self.ids_map.append(global_idx)
                global_idx += 1
                
                if global_idx % 100 == 0:
                    print(f"Processed {global_idx}", end="\r")
                    
        self.dense_matrix = np.vstack(self.dense_index)
        print(f"\nIndexing complete. Total: {len(self.ids_map)}")

    def _search_dense(self, query_vec: List[float], top_k: int) -> List[int]:
        q_vec = self._normalize_dense(query_vec)
        scores = np.dot(self.dense_matrix, q_vec)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [self.ids_map[i] for i in top_indices]
    
    def _search_sparse(self, query_sparse: Dict[int, float], top_k: int) -> List[int]:
        scores = []
        for idx, doc_sparse in enumerate(self.sparse_index):
            score = 0.0
            if len(query_sparse) < len(doc_sparse):
                for token, weight in query_sparse.items():
                    if token in doc_sparse:
                        score += weight * doc_sparse[token]
            else:
                for token, weight in doc_sparse.items():
                    if token in query_sparse:
                        score += weight * query_sparse[token]
            
            if score > 0:
                scores.append((self.ids_map[idx], score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in scores[:top_k]]

    def search(self, query: str, limit: int) -> List[Dict]:
        q_dense, q_sparse = self.embedder.embed(query)
        
        if not self.hybrid_embedder:
            q_sparse = {}
            for t in self._tokenize(query):
                if t in self.bm25_model.idf:
                    q_sparse[self._hash_token(t)] = 1.0

        dense_ids = self._search_dense(q_dense, limit * 2)
        sparse_ids = self._search_sparse(q_sparse, limit * 2)
        
        fused_results = self.rrf.fuse([dense_ids, sparse_ids])[:limit]
        
        output = []
        for doc_id, rrf_score in fused_results:
            doc_data = self.docs_store[doc_id]
            output.append({
                "rrf_score": rrf_score,
                "source": doc_data["source"],
                "data": doc_data["content"]
            })
            
        return output