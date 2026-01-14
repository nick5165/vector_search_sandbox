import numpy as np
from rank_bm25 import BM25Okapi
from collections import defaultdict, Counter
from typing import Tuple, List, Dict, Any, Optional

from text_utils import tokenize_text, split_into_sentences
from embedders import BaseEmbedder

class BaseRetriever:
    """
    Базовый абстрактный класс для всех модулей поиска.
    Определяет единый интерфейс для индексации и поиска.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.doc_map = {}
        self.doc_ids = []
        
    def index(self, documents: List[Dict], granularity: str = "doc"):
        """
        Создает индекс на основе переданных документов.
        """
        pass
    
    def search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """
        Выполняет поиск по индексу.
        """
        pass
    
class DenseRetriever(BaseRetriever):
    """
    Реализует семантический поиск.
    """
    
    def __init__(self, embedder: BaseEmbedder, name: str = "dense"):
        super().__init__(name)
        self.embedder = embedder
        self.vectors = []
        self.sparse_vectors = []
        self.index_to_doc_id = []
        self.matrix = None
        self.has_sparse_data = False
        
    def _normalize(self, vec):
        """
        Нормализация вектора для Cosine Similarity.
        """
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
    
    def index(self, documents: List[Dict], granularity: str = "doc"):
        """
        Векторизует документы. Сохраняет и Dense, и Sparse представления.
        """
        
        print(f"[{self.name}] Indexing {len(documents)} docs (mode={granularity})...")
        self.vectors = []
        self.sparse_vectors = []
        self.index_to_doc_id = []
        
        for doc_idx, doc in enumerate(documents):
            text = doc.get("chunk_text") or doc.get("question") or ""
            
            texts_to_embed = []
            
            if granularity == "sentence":
                sents = split_into_sentences(text)
                if not sents:
                    sents = [text]
                texts_to_embed = sents
            else:
                texts_to_embed = [text]
                
            for t in texts_to_embed:
                vec = self.embedder.embed_dense(t)
                self.vectors.append(self._normalize(vec))

                sp_vec = self.embedder.embed_sparse(t)
                self.sparse_vectors.append(sp_vec)
                
                self.index_to_doc_id.append(doc_idx)
            
        if self.vectors:    
            self.matrix = np.vstack(self.vectors)

            self.has_sparse_data = any(len(d) > 0 for d in self.sparse_vectors)
            if self.has_sparse_data:
                print(f"[{self.name}] Sparse vectors detected. Hybrid search enabled.")
        else:
            self.matrix = np.array([])
            
    def search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """
        Поиск ближайших соседей.
        Score = DotProduct(Dense) + DotProduct(Sparse).
        """

        q_vec = self._normalize(self.embedder.embed_dense(query))
        dense_scores = np.dot(self.matrix, q_vec)
        
        final_scores = dense_scores

        if self.has_sparse_data:
            q_sparse = self.embedder.embed_sparse(query)

            if q_sparse:
                combined_scores = np.copy(dense_scores)

                for i, doc_sparse in enumerate(self.sparse_vectors):
                    if not doc_sparse:
                        continue
                        
                    sparse_score = 0.0
                    for token_id, q_weight in q_sparse.items():
                        if token_id in doc_sparse:
                            sparse_score += q_weight * doc_sparse[token_id]
                    
                    combined_scores[i] += sparse_score
                
                final_scores = combined_scores

        top_indices = np.argsort(final_scores)[-top_k*5:][::-1]
        
        doc_scores = {}
        for idx in top_indices:
            real_doc_id = self.index_to_doc_id[idx]
            score = float(final_scores[idx])
            
            if real_doc_id not in doc_scores or score > doc_scores[real_doc_id]:
                doc_scores[real_doc_id] = score
                
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return sorted_docs
    
class BM25Retriever(BaseRetriever):
    """
    Реализует лексический поиск (по ключевым словам) с использованием алгоритма BM25.
    """
    def __init__(self, name: str = "bm25"):
        super().__init__(name)
        self.bm25 = None
        self.corpus_index_to_doc_id = []
        
    def index(self, documents: List[Dict], granularity: str = "doc"):
        """
        Токенизирует документы и строит обратный индекс BM25.
        """
        
        print(f"[{self.name}] Indexing {len(documents)} docs (mode={granularity})...")
        tokenized_corpus = []
        self.corpus_index_to_doc_id = []
        
        for doc_idx, doc in enumerate(documents):
            text = doc.get("chunk_text") or doc.get("question") or ""
            
            segments = []
            if granularity == "sentence":
                sents = split_into_sentences(text)
                segments = sents if sents else [text]
            else:
                segments = [text]
                
            for seg in segments:
                tokens = tokenize_text(seg)
                
                if tokens:
                    tokenized_corpus.append(tokens)
                    self.corpus_index_to_doc_id.append(doc_idx)
                    
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"[{self.name}] Trained on {len(tokenized_corpus)} items (segments).")
        
    def search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """
        Выполняет поиск по ключевым словам.
        """
        
        tokenized_query = tokenize_text(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_n_indices = np.argsort(scores)[-top_k*5:][::-1]
        seen_docs = {}
        
        for idx in top_n_indices:
            score = float(scores[idx])
            
            if score <= 0:
                continue
            
            real_doc_id = self.corpus_index_to_doc_id[idx]
            
            if real_doc_id not in seen_docs:
                seen_docs[real_doc_id] = score
            
            if len(seen_docs) >= top_k:
                break
            
        results = list(seen_docs.items()) 
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
class EnsembleRetriever:
    """
    Класс-оркестратор для гибридного поиска.
    Объединяет результаты нескольких ретриверов через RRF.
    """
    
    def __init__(self, retrievers: List[BaseRetriever]):
        self.retrievers = retrievers
        self.documents_store = {}
        
    def set_documents(self, docs: List[Dict]):
        """
        Сохраняет ссылку на исходные документы.
        """
        self.documents_store = {i: d for i, d in enumerate(docs)}
        
    def rrf_fuse(self, results_list: List[List[Tuple[int, float]]], k: int = 60):
        """
        Реализация Reciprocal Rank Fusion.
        """
        
        rrf_map = defaultdict(float)
        
        for ranking in results_list:
            for rank, (doc_id, score) in enumerate(ranking):
                rrf_map[doc_id] += 1.0 / (k + rank + 1)
        
        sorted_ids = sorted(rrf_map.items(), key=lambda x: x[1], reverse=True)
        
        final_output = []
        for doc_id, score in sorted_ids:
            if doc_id in self.documents_store:
                final_output.append({
                    "rrf_score": score,
                    "doc_id": doc_id,
                    "content": self.documents_store[doc_id]
                })
        return final_output

    def search(self, query: str, limit: int = 30) -> List[Dict]:
        """
        Главный метод поиска.
        """
        all_results = []
        for r in self.retrievers:
            all_results.append(r.search(query, top_k=limit * 2))
            
        fused = self.rrf_fuse(all_results)[:limit]
        return fused