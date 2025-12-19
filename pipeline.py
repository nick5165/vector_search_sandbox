import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pandas as pd
import sys
from tqdm import tqdm

from database import JSONLDataset, XLSXDataset 
from embedders import BGEHybridEmbedder, FastTextEmbedder
from retrieval import DenseRetriever, BM25Retriever, EnsembleRetriever

PATH_FASTTEXT = "/home/mikhailovnk/run_git/vector_search_sandbox/cc.ru.300.bin"
PATH_BGE = "/home/mikhailovnk/00_Models/Embedders/BGE-M3"
PATH_DOCS = "/home/mikhailovnk/run_git/vector_search_sandbox/indorcad.jsonl"
PATH_FAQ = "/home/mikhailovnk/run_git/vector_search_sandbox/faq.jsonl"
PATH_QUERIES = "/home/mikhailovnk/run_git/vector_search_sandbox/Ответы_IndorAssistant_v2.xlsx"
OUTPUT_DIR = "./comparison_results"

def load_data():
    """
    Загружает документы. Если документов нет — возвращает пустой список.
    """
    docs = []
    sources = [PATH_DOCS, PATH_FAQ]
    
    for path in sources:
        if os.path.exists(path):
            print(f"Loading from {path}...")
            ds = JSONLDataset(path) 
            
            for i in tqdm(range(len(ds)), desc=f"Reading {os.path.basename(path)}"):
                rec = ds[i]
                rec['source_file'] = os.path.basename(path)
                docs.append(rec)
        else:
            print(f"Warning: Path not found {path}")
            
    return docs

def load_queries():
    """
    Загружает вопросы из Excel.
    Если файла нет или колонка не найдена — вызывает ошибку (никаких заглушек).
    """
    if not os.path.exists(PATH_QUERIES):
        print(f"КРИТИЧЕСКАЯ ОШИБКА: Файл с вопросами не найден: {PATH_QUERIES}")
        sys.exit(1)

    print(f"Loading queries from {PATH_QUERIES}...")
    ds = XLSXDataset(PATH_QUERIES)
    
    target_col = None
    for col in ds.columns:
        if 'Вопрос' in str(col):
            target_col = col
            break
    
    if target_col:
        print(f"Found query column: '{target_col}'")
        return ds.df[target_col].dropna().astype(str).tolist()
    else:
        print("КРИТИЧЕСКАЯ ОШИБКА: В Excel файле не найдена колонка 'Вопрос' или 'Query'.")
        sys.exit(1)

def save_report_xlsx(results_map, filename="report.xlsx"):
    rows = []
    if not results_map:
        print("No results to save.")
        return

    first_q = next(iter(results_map))
    methods = list(results_map[first_q].keys())
    
    for query, methods_data in tqdm(results_map.items(), desc="Generating Report"):
        max_hits = max(len(hits) for hits in methods_data.values())
        
        for rank in range(max_hits):
            row = {
                "Query": query,
                "Rank": rank + 1
            }
            
            for method_name in methods:
                hits = methods_data.get(method_name, [])
                
                if rank < len(hits):
                    hit = hits[rank]
                    content = hit['content']

                    text = content.get("chunk_text") or content.get("answer") or content.get("question") or ""
                    source = content.get("source_file", "")

                    score = hit.get('rrf_score', 0)

                    row[f"{method_name} Score"] = round(score, 4)
                    row[f"{method_name} Text"] = text[:32000]
                    row[f"{method_name} Source"] = source
                else:
                    row[f"{method_name} Score"] = None
                    row[f"{method_name} Text"] = ""
                    row[f"{method_name} Source"] = ""
            
            rows.append(row)

    df = pd.DataFrame(rows)
    
    cols = ["Query", "Rank"]
    for m in methods:
        cols.extend([f"{m} Score", f"{m} Text", f"{m} Source"])
    
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, filename)
    
    print(f"Saving Excel to {out_path}...")
    try:
        df.to_excel(out_path, index=False, engine='openpyxl')
        print("Done.")
    except Exception as e:
        print(f"Error saving Excel: {e}")


def main():
    all_docs = load_data()
    if not all_docs:
        print("CRITICAL: No documents loaded. Exiting.")
        sys.exit(1)
    print(f"Total documents: {len(all_docs)}")

    print("Loading models...")
    try:
        bge = BGEHybridEmbedder(PATH_BGE, device="cpu")
    except Exception as e:
        print(f"Error loading BGE: {e}")
        sys.exit(1)

    try:
        ft = FastTextEmbedder(PATH_FASTTEXT)
    except Exception as e:
        print(f"Error loading FastText: {e}")
        sys.exit(1)

    print("\n--- Starting Indexing ---")
    
    ft_retriever = DenseRetriever(ft, name="FastText_Doc")
    ft_retriever.index(tqdm(all_docs, desc="Indexing FastText"), granularity="doc")

    bge_retriever_doc = DenseRetriever(bge, name="BGE_Doc")
    bge_retriever_doc.index(tqdm(all_docs, desc="Indexing BGE (CPU)"), granularity="doc")

    bm25_retriever = BM25Retriever(name="BM25")
    bm25_retriever.index(tqdm(all_docs, desc="Indexing BM25"))

    p1 = EnsembleRetriever([bge_retriever_doc])
    p1.set_documents(all_docs)

    p2 = EnsembleRetriever([ft_retriever, bm25_retriever])
    p2.set_documents(all_docs)

    p3 = EnsembleRetriever([bge_retriever_doc, ft_retriever, bm25_retriever])
    p3.set_documents(all_docs)

    queries = load_queries()
    
    if not queries:
        print("Список вопросов пуст. Выход.")
        sys.exit(0)

    queries_to_run = queries[:20] 
    
    comparison_data = {}
    limit_results = 5
    
    print(f"\nRunning comparison on {len(queries_to_run)} queries...")
    
    for q in tqdm(queries_to_run, desc="Processing Queries"):
        q = str(q).strip()
        if not q: continue
        
        res_p1 = p1.search(q, limit=limit_results)
        res_p2 = p2.search(q, limit=limit_results)
        res_p3 = p3.search(q, limit=limit_results)
        
        comparison_data[q] = {
            "BGE_Only": res_p1,
            "FastText_BM25": res_p2,
            "All_Combined": res_p3
        }

    save_report_xlsx(comparison_data, "comparison_matrix.xlsx")

if __name__ == "__main__":
    main()