import random
from database import XLSXDataset
from retrieve import Retriever
from embedders import BGEHyibridEmbedder

PATH_TO_MODEL = ""
PATH_TO_DOCS = ""
PATH_TO_FAQ = ""
PATH_TO_QUERIES = ""

embedder = BGEHyibridEmbedder(PATH_TO_MODEL) 
engine = Retriever(embedder)

files_map = {
    "documentation": PATH_TO_DOCS,
    "faq": PATH_TO_FAQ
}
engine.index_datasets(files_map)

questions_path = PATH_TO_QUERIES
questions_dataset = XLSXDataset(questions_path)

random_idx = random.randint(0, len(questions_dataset) - 1)
item = questions_dataset[random_idx]

query_text = item.get("Вопрос", "Текст вопроса не найден")

print(f"\n>>> Вопрос из Excel (строка {random_idx+2}):")
print(f"'{query_text}'")
print("-" * 50)

results = engine.search(str(query_text), limit=3)

for res in results:
    print(f"Score: {res['rrf_score']:.4f} | Src: {res['source']}")
    content = res['data'].get('chunk_text', str(res['data']))
    print(f"Text: {content[:150]}...")
    print("-" * 50)