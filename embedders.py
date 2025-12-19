import torch
import fasttext
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from abc import ABC, abstractmethod
from typing import Any, Tuple, List, Dict

class BaseEmbedder(ABC):
    """
    Абстрактный интерфейс сервисов генерации эмбеддингов.
    """
    
    @abstractmethod
    def embed_dense(self, text: str) -> np.ndarray:
        pass
    
    @abstractmethod
    def embed_sparse(self, text: str) -> Dict[int, float]:
        pass

class BGEHybridEmbedder(BaseEmbedder):
    def __init__(self, model_path: str, device:str = None):
        """
        Инициализация модели.
        model_path: путь к локальной папке или имя модели на HuggingFace.
        device: 'cpu' или 'cuda'. Если None, выбирается автоматически.
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
                
        print(f"BGE-M3 Loading on {device}")
        self.model = BGEM3FlagModel(
            model_name_or_path = model_path,
            use_fp16 = (device == "cuda"),
            device = device
        )
        self.tokenizer = self.model.tokenizer
        
    def embed_dense(self, text) -> np.ndarray:
        output = self.model.encode(text, return_dense=True, return_sparse=False)
        return output['dense_vecs'].tolist()
    
    def embed_sparse(self, text):
        output = self.model.encode(text, return_dense=False, return_sparse=True)
        raw_weights = output["lexical_weights"]
        
        sparse_vec = {}
        
        for token, weight in raw_weights.items():
            tid = self.tokenizer.convert_tokens_to_ids(token)
            sparse_vec[tid] = float(weight)
        return sparse_vec
    
class FastTextEmbedder(BaseEmbedder):
    def __init__(self, model_path: str):
        """
        model_path: путь к файлу .bin 
        """
        print(f"Loading FastText model from {model_path}...")
        self.model = fasttext.load_model(model_path)
        
    def embed_dense(self, text) -> np.ndarray:
        text = text.replace("\n", " ")
        return np.array(self.model.get_sentence_vector(text))
    
    def embed_sparse(self, text) -> Dict[int, float]:
        return {}