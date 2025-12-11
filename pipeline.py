import random
from database import XLXSDataset 
from retrieve import Retriever
from embedders import BGEHyibridEmbedder

PATH_TO_MODEL = ""
PATH_TO_DOCS = ""
PATH_TO_FAQ = ""
PATH_TO_QUERIES = ""

embedder = BGEHyibridEmbedder() 
engine = Retriever(embedder)

files_map = {
    "documentation": "docs_dump.jsonl",
    "faq": "faq_dump.jsonl"
}
engine.index_datasets(files_map)

questions_path = "questions.xlsx"
questions_dataset = XLXSDataset(questions_path)

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