import torch
from FlagEmbedding import BGEM3FlagModel

class BGEHyibridEmbedder:
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
            model_path = model_path,
            use_fp16 = (device == "cuda"),
            device = device
        )
        self.tokenizer = self.model.tokenizer
        
    def embed(self, text: str):
        """
        Возвращает Dense и Sparse эмбеддинги для одного текста.
        """
        output = self.model.encode(
            text,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False
        )
        dense_vec = output["dense_vecs"].tolist()
        
        raw_sparse = output["lexical_weights"]
        sparse_vec = {}
        
        for token_str, weight in raw_sparse.items():
            token_id = self.tokenizer.convert_tokens_to_ids(token_str)
            sparse_vec[token_id] = weight
            
        return dense_vec, sparse_vec