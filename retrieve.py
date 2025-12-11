import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Any

from database import JSONLDataset

class RRFProcessor:
    """
    Реализация алгоритма Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, k: int = 100):
        self.k = k
        
    def fuse(self, ranked_lists: List[List[int]]) ->List[Tuple[int, float]]:
        """
        Объединяет несколько списков ранжирования.
        :param ranked_lists: Список списков ID документов [[id1, id2...], [id3, id1...]]
        :return: Список кортежей (doc_id, score), отсортированный по score.
        """
        rrf_scores = defaultdict(float)
        for ranking in ranked_lists:
            for rank, doc_id in enumerate(ranking):
                rrf_scores[doc_id] += 1.0/(self.k + rank + 1)
                
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_results

class Retriever:
    """
    Эмулятор работы Milvus in-memory.
    Поддерживает Dense (Cosine/IP) и Sparse (IP) поиск.
    """
    def __init__(self, embedder):
        self.embedder = embedder
        self.rrf = RRFProcessor()
        
        self.docs_store: Dict[int, Dict[str, Any]] = {}
        
        self.dense_index = []
        self.sparse_index: List[Dict[int, float]] = []
        
        self.ids_map: List[int] = []
        
    def _normalize_dense(self, vector: List[float]) -> np.ndarray:
        """
        Нормализация вектора для Cosine Similarity через Dot Product
        """
        arr = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            return arr / norm
        return arr
    
    def index_datasets(self, datasets_paths: Dict[str, str]):
        """
        Загружает и индексирует базы данных.
        :param datasets_paths: словарь {'название_источника': 'путь_к_файлу.jsonl'}
        """
        global_idx = 0
        for source_name, path in datasets_paths.items():
            print(f"Indexing {source_name} from {path}")
            dataset = JSONLDataset(path)
            
            for i in range(len(dataset)):
                record = dataset[i]
                
                text_to_embed = record.get("chunk_text") or record.get('question')
                
                dense_vec, sparse_vec = self.embedder.embed(text_to_embed)
                self.dense_index.append(self._normalize_dense(dense_vec))
                
                self.sparse_index.append(sparse_vec)
                
                self.docs_store[global_idx] = {
                    "source": source_name,
                    "content": record
                }
                self.ids_map.append(global_idx)
                global_idx += 1
                
                if global_idx % 50 == 0:
                    print(f"Processed {global_idx} chunks", end="\r")
                    
        self.dense_matrix = np.vstack(self.dense_index)
        print(f"\nEnd of indexing. Total: {len(self.ids_map)}")
        
    def _search_dense(self, query_vec: List[float], top_k: int) -> List[int]:
        """
        Аналог поиска Milvus с metric_type="COSINE" (на нормализованных векторах это IP).
        """
        q_vec = self._normalize_dense(query_vec)
        scores = np.dot(self.dense_matrix, q_vec)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        return [self.ids_map[i] for i in top_indices]
    
    def _search_sparse(self, query_sparse: Dict[int, float], top_k: int) -> List[int]:
        """
        Аналог поиска Milvus с metric_type="IP" (Inner Product) для SPARSE_FLOAT_VECTOR.
        Формула: sum(q_weight * d_weight) для совпадающих токенов.
        """
        scores = []
        
        for idx, doc_sparse in enumerate(self.sparse_index):
            score = 0.0
            
            if len(query_sparse)< len(doc_sparse):
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
        """
        Гибридный поиск:
        1. Dense Search
        2. Sparse Search (Milvus style IP)
        3. RRF Fusion
        """
        q_dense, q_sparse = self.embedder.embed(query)
        
        dense_ids = self._search_dense(q_dense, limit * 2)
        sparse_ids = self._search_sparse(q_sparse, limit * 2)
        
        fused_results = self.rrf.fuse([dense_ids, sparse_ids], limit=limit)
        output = []
        for doc_id, rrf_score in fused_results:
            doc_data = self.docs_store[doc_id]
            output.append({
                "rrf_score": rrf_score,
                "source": doc_data["source"],
                "data": doc_data["content"]
            })
            
        return output