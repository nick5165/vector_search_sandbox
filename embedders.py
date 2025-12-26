import torch
import gensim.models.keyedvectors
import gensim.models.deprecated.keyedvectors
import fasttext
import numpy as np
from abc import ABC, abstractmethod
from FlagEmbedding import BGEM3FlagModel
from typing import Any, Tuple, List, Dict
from text_utils import tokenize_text

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
    
class RusVectoresEmbedder(BaseEmbedder):
    def __init__(self, model_path: str):
        print(f"Loading RusVectores using Gensim {gensim.__version__} from {model_path}...")

        if not hasattr(gensim.models.deprecated.keyedvectors, 'FastTextKeyedVectors'):
            setattr(gensim.models.deprecated.keyedvectors, 
                    'FastTextKeyedVectors', 
                    gensim.models.keyedvectors.FastTextKeyedVectors)

        try:
            self.model = gensim.models.FastText.load(model_path)
            self.wv = self.model.wv
        except Exception as e:
            print(f"FastText load failed ({e}), trying KeyedVectors...")
            self.wv = gensim.models.KeyedVectors.load(model_path)
            
        self.vector_size = self.wv.vector_size
        print(f"Model loaded. Vector size: {self.vector_size}")

    def embed_dense(self, text: str) -> np.ndarray:
        tokens = tokenize_text(text)
        vectors = []
        for token in tokens:
            if token in self.wv:
                vectors.append(self.wv[token])
            
        if not vectors:
            return np.zeros(self.vector_size)
        
        return np.array(np.mean(vectors, axis=0))

    def embed_sparse(self, text: str):
        return {}