import pandas as pd

class BasePandasDataset:
    """
    Базовый класс для датасетов на основе Pandas
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.data = df.to_dict(orient='records')
        self.columns = df.columns.to_list()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if isinstance(idx, str):
            if idx in self.df.columns:
                return self.df[idx].to_list()
            else:
                raise KeyError(f"Column {idx} not found in dataset")
            
        elif isinstance(idx, slice):
            return self.data[idx]
        
        return self.data[idx]
    
class XLSXDataset(BasePandasDataset):
    def __init__(self, path: str):
        print(f"Loading XLSX from {path}...")
        df = pd.read_excel(path)
        super().__init__(df)
        
class JSONLDataset(BasePandasDataset):
    def __init__(self, path: str):
        print(f"Loading JSONL from {path}...")
        df = pd.read_json(path, lines=True)
        super().__init__(df)