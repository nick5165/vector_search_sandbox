import numpy as np
from rank_bm25 import BM25Okapi
from collections import defaultdict
from typing import Tuple, List, Dict

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
    Реализует семантический поиск с гибридным ранжированием (RRF) 
    и оптимизацией через Inverted Index.
    """
    
    def __init__(self, embedder: BaseEmbedder, name: str = "dense"):
        super().__init__(name)
        self.embedder = embedder
        self.vectors = []
        self.sparse_vectors = [] 

        self.inverted_index = defaultdict(list)
        
        self.index_to_doc_id = []
        self.matrix = None
        self.has_sparse_data = False
        
    def _normalize(self, vec):
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
    
    def index(self, documents: List[Dict], granularity: str = "doc"):
        """
        Векторизует документы и строит обратный индекс (Inverted Index).
        """
        print(f"[{self.name}] Indexing {len(documents)} docs (mode={granularity})...")

        self.vectors = []
        self.sparse_vectors = []
        self.index_to_doc_id = []
        self.inverted_index = defaultdict(list)
        
        for doc_idx, doc in enumerate(documents):
            text = doc.get("chunk_text") or doc.get("question") or ""
            
            texts_to_embed = []
            if granularity == "sentence":
                sents = split_into_sentences(text)
                if not sents: sents = [text]
                texts_to_embed = sents
            else:
                texts_to_embed = [text]
                
            for t in texts_to_embed:
                vec = self.embedder.embed_dense(t)
                self.vectors.append(self._normalize(vec))

                sp_vec = self.embedder.embed_sparse(t)
                self.sparse_vectors.append(sp_vec)

                current_internal_id = len(self.vectors) - 1

                if sp_vec:
                    for token_id, weight in sp_vec.items():
                        if weight > 0:
                            self.inverted_index[token_id].append((current_internal_id, weight))
                
                self.index_to_doc_id.append(doc_idx)
            
        if self.vectors:    
            self.matrix = np.vstack(self.vectors)
            self.has_sparse_data = bool(self.inverted_index)
            if self.has_sparse_data:
                print(f"[{self.name}] Sparse vectors detected. Inverted Index built.")
        else:
            self.matrix = np.array([])

    def search(self, query: str, top_k: int, k_const: int = 60) -> List[Tuple[int, float]]:
        """
        Поиск с использованием RRF и ускоренный через Inverted Index.
        """

        q_vec = self._normalize(self.embedder.embed_dense(query))
        dense_scores = np.dot(self.matrix, q_vec)
        
        candidates_limit = top_k * 2
        dense_top_indices = np.argsort(dense_scores)[-candidates_limit:][::-1]

        sparse_top_indices = []
        
        if self.has_sparse_data:
            q_sparse = self.embedder.embed_sparse(query)
            
            if q_sparse:
                sparse_scores = np.zeros(self.matrix.shape[0])
                
                for token_id, q_weight in q_sparse.items():
                    if token_id in self.inverted_index:
                        postings = self.inverted_index[token_id]

                        for doc_idx, doc_weight in postings:
                            sparse_scores[doc_idx] += q_weight * doc_weight
                
                sparse_top_indices = np.argsort(sparse_scores)[-candidates_limit:][::-1]

        rrf_map = {} 

        def add_ranks(indices):
            for rank, idx in enumerate(indices):
                real_doc_id = self.index_to_doc_id[idx]
                score = 1.0 / (k_const + rank + 1)
                rrf_map[real_doc_id] = rrf_map.get(real_doc_id, 0.0) + score

        add_ranks(dense_top_indices)
        
        if self.has_sparse_data and len(sparse_top_indices) > 0:
            add_ranks(sparse_top_indices)

        sorted_docs = sorted(rrf_map.items(), key=lambda x: x[1], reverse=True)[:top_k]
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