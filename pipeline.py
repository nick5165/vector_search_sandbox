import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from database import XLSXDataset, JSONLDataset
from retrieve import Retriever
from embedders import BGEHybridEmbedder, FastTextEmbedder

PATH_TO_MODEL = "/home/mikhailovnk/run_git/vector_search_sandbox/cc.ru.300.bin" 
PATH_TO_DOCS = "/home/mikhailovnk/run_git/vector_search_sandbox/indorcad.jsonl"
PATH_TO_FAQ = "/home/mikhailovnk/run_git/vector_search_sandbox/faq.jsonl"
PATH_TO_QUERIES = "/home/mikhailovnk/run_git/vector_search_sandbox/Ответы_IndorAssistant_v2.xlsx"

def main():
    print(">>> Загрузка модели и индексация...")
    
    if not os.path.exists(PATH_TO_DOCS) and not os.path.exists(PATH_TO_FAQ):
        print("ОШИБКА: Файлы базы знаний не найдены.")
        return

    embedder = FastTextEmbedder(PATH_TO_MODEL)
    engine = Retriever(embedder, hybrid_embedder=False)

    files_map = {
        "documentation": PATH_TO_DOCS,
        "faq": PATH_TO_FAQ
    }
    engine.index_datasets({k: v for k, v in files_map.items() if os.path.exists(v)})

    questions_dataset = None
    max_idx = -1
    
    if os.path.exists(PATH_TO_QUERIES):
        print(f">>> Чтение файла вопросов: {PATH_TO_QUERIES}")
        try:
            questions_dataset = XLSXDataset(PATH_TO_QUERIES)
            max_idx = len(questions_dataset) - 1
            print(f"    Загружено строк: {len(questions_dataset)}")
            if len(questions_dataset) > 0:
                print(f"    Поля: {list(questions_dataset[0].keys())}")
        except Exception as e:
            print(f"    Ошибка чтения Excel: {e}")
    else:
        print(">>> Файл с вопросами не найден. Доступен только ручной ввод.")

    while True:
        print("\n" + "="*60)
        if questions_dataset:
            print(f"1. Выбрать вопрос по индексу (0 - {max_idx})")
        else:
            print("1. (Недоступно - файл не загружен)")
        print("2. Ввести вопрос вручную")
        print("q. Выход")
        
        choice = input("Выбор: ").strip().lower()
        query_text = ""

        if choice == '1':
            if not questions_dataset:
                print("Датасет недоступен.")
                continue
                
            try:
                idx_str = input(f"Введите номер строки (0-{max_idx}): ")
                idx = int(idx_str)
                
                if not (0 <= idx <= max_idx):
                    print(f"Ошибка: индекс вне диапазона.")
                    continue

                item = questions_dataset[idx]
                query_text = item.get("Вопрос") or item.get("question") or item.get("text")
                
                if not query_text:
                    print(f"В строке {idx} пустое поле вопроса. Данные: {item}")
                    continue
                
                print(f"\n>>> Вопрос [idx={idx}]:\n'{query_text}'")
                
            except ValueError:
                print("Ошибка: введите целое число.")
                continue

        elif choice == '2':
            query_text = input("\n>>> Введите запрос: ").strip()
            if not query_text: continue

        elif choice == 'q':
            break
        else:
            continue

        print("-" * 60)
        results = engine.search(str(query_text), limit=5)

        if not results:
            print("Ничего не найдено.")
        
        for i, res in enumerate(results, 1):
            score = res.get('rrf_score', 0.0)
            src = res.get('source', 'unknown')
            data = res.get('data', {})
            
            content = data.get('chunk_text') or data.get('answer') or str(data)
            content = str(content).replace('\n', ' ')
            
            print(f"#{i} [Score: {score:.4f}] [{src}]")
            print(f"{content}")
            print("-" * 30)

if __name__ == "__main__":
    main()