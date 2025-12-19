import zlib
import numpy as np
from rank_bm25 import BM25Okapi
from collections import defaultdict, Counter
from typing import Tuple, List, Dict, Any, Optional

from embedders import BaseEmbedder
from text_utils import tokenize_text, split_into_sentences

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
        
        Args:
            documents: Список словарей с данными документов.
            granularity: Уровень детализации ('doc' - документ целиком, 'sentence' - по предложениям).
        """
        
        pass
    
    def search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """
        Выполняет поиск по индексу.
        
        Args:
            query: Текст запроса.
            top_k: Количество возвращаемых результатов.
            
        Returns:
            Список кортежей (doc_id, score), отсортированный по убыванию релевантности.
        """
        
        pass
    
class DenseRetriever(BaseRetriever):
    """
    Реализует семантический поиск на основе плотных векторов (Dense Vectors).
    Использует переданный embedder (FastText, BGE, BERT и т.д.).
    """
    
    def __init__(self, embedder: BaseEmbedder, name: str = "dense"):
        super().__init__(name)
        self.embedder = embedder
        self.vectors = []
        self.index_to_doc_id = []
        
    def _normalize(self, vec):
        """
        Нормализация вектора.
        Необходима для того, чтобы Dot Product был эквивалентен Cosine Similarity.
        """
        
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
    
    def index(self, documents: List[Dict], granularity: str = "doc"):
        """
        Векторизует документы и собирает их в матрицу.
        
        Логика granularity:
        - 'doc': Векторизуется весь текст поля chunk_text/question.
        - 'sentence': Текст разбивается на предложения, каждое векторизуется отдельно. 
          В индекс попадает много векторов, ссылающихся на один real_doc_id.
        """
        
        print(f"[{self.name}] Indexing {len(documents)} docs (mode={granularity})...")
        self.vectors = []
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
                self.index_to_doc_id.append(doc_idx)
                
            self.matrix = np.vstack(self.vectors)
            
    def search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """
        Поиск ближайших соседей через скалярное произведение (Dot Product).
        Реализует стратегию 'Max Score Aggregation': если документ разбит на предложения,
        берется скор самого релевантного предложения.
        """
        
        q_vec = self._normalize(self.embedder.embed_dense(query))
        scores = np.dot(self.matrix, q_vec)
        
        top_indices = np.argsort(scores)[-top_k*3:][::-1]
        
        doc_scores = {}
        for idx in top_indices:
            real_doc_id = self.index_to_doc_id[idx]
            score = float(scores[idx])
            
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
        
        При granularity='sentence' каждое предложение считается отдельным "документом" 
        для статистики BM25, но при поиске результаты схлопываются обратно в исходный документ.
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
        Агрегирует результаты: если найдено несколько сегментов одного документа, 
        берется лучший (по аналогии с DenseRetriever).
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
    Объединяет результаты нескольких ретриверов (Dense, BM25) 
    используя алгоритм Reciprocal Rank Fusion (RRF).
    """
    
    def __init__(self, retrievers: List[BaseRetriever]):
        self.retrievers = retrievers
        self.documents_store = {}
        
    def set_documents(self, docs: List[Dict]):
        """
        Сохраняет ссылку на исходные документы, чтобы возвращать их контент в search().
        """
        
        self.documents_store = {i: d for i, d in enumerate(docs)}
        
    def rrf_fuse(self, results_list: List[List[Tuple[int, float]]], k: int = 60):
        """
        Реализация Reciprocal Rank Fusion.
        Score = sum(1 / (k + rank_i + 1)) для каждого ретривера.
        
        Args:
            results_list: Список списков результатов от разных ретриверов.
            k: Константа сглаживания (обычно 60).
            
        Returns:
            Список словарей с финальным скором и контентом документов.
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

    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Главный метод поиска.
        1. Запрашивает результаты у всех дочерних ретриверов.
        2. Объединяет их через RRF.
        3. Возвращает топ-limit документов с контентом.
        """
        all_results = []
        for r in self.retrievers:
            all_results.append(r.search(query, top_k=limit * 2))
            
        fused = self.rrf_fuse(all_results)[:limit]
        return fused