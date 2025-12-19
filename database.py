import pandas as pd

class XLSXDataset:
    """
    Датасет для чтения Excel-файлов, возвращающий данные с названиями колонок.
    """
    def __init__(self, path: str):
        self.path = path
        self.df = pd.read_excel(self.path)
        
        self.columns = self.df.columns.tolist()
        
        self.data = self.df.values.tolist()
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return dict(zip(self.columns, self.data[idx]))


class JSONLDataset:
    """
    Датасет для формата JSON Lines (.jsonl).
    """
    def __init__(self, path):
        self.df = pd.read_json(path, lines=True)
        self.data = self.df.to_dict(orient='records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]