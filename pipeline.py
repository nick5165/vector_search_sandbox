import os
import sys
import pandas as pd
from tqdm import tqdm
from typing import List, Dict
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from database import JSONLDataset, XLSXDataset
from embedders import BGEHybridEmbedder, FastTextEmbedder
from retrieval import DenseRetriever, BM25Retriever, EnsembleRetriever

PATHS = {
    "fasttext": "/home/mikhailovnk/00_Models/Embedders/cc.ru.300.bin",
    "bge": "/home/mikhailovnk/00_Models/Embedders/BGE-M3",
    "docs": [
        "indorcad.jsonl",
        "faq.jsonl"
    ],
    "queries": "/home/mikhailovnk/run_git/vector_search_sandbox/Ответы_IndorAssistant_v2.xlsx",
    "output_dir": "./comparison_results"
}
REPORT_FILENAME = "comparison_matrix.xlsx"
TOP_K_RESULTS = 5
QUERIES_LIMIT = 20  

def load_documents(file_paths: List[str]) -> List[Dict]:
    docs = []
    for path in file_paths:
        if not os.path.exists(path):
            print(f"Warning: Path not found {path}")
            continue
            
        print(f"Loading from {os.path.basename(path)}...")
        ds = JSONLDataset(path)
        for i in tqdm(range(len(ds)), desc="Reading records"):
            rec = ds[i]
            rec['source_file'] = os.path.basename(path)
            docs.append(rec)
    return docs

def load_queries(path: str) -> List[str]:
    if not os.path.exists(path):
        print(f"CRITICAL: Queries file not found: {path}")
        sys.exit(1)

    print(f"Loading queries from {path}...")
    ds = XLSXDataset(path)
    
    target_col = next((col for col in ds.columns if 'Вопрос' in str(col)), None)
    
    if not target_col:
        print("CRITICAL: Column 'Вопрос' not found in Excel.")
        sys.exit(1)
        
    return [str(q).strip() for q in ds[target_col] if str(q).strip()]

def save_report(results_map: Dict, output_path: str):
    if not results_map:
        print("No results to save.")
        return

    rows = []
    first_q = next(iter(results_map))
    methods = list(results_map[first_q].keys())

    for query, methods_data in tqdm(results_map.items(), desc="Generating Report"):
        max_hits = max((len(hits) for hits in methods_data.values()), default=0)
        
        for rank in range(max_hits):
            row = {"Query": query, "Rank": rank + 1}
            
            for method in methods:
                hits = methods_data.get(method, [])
                if rank < len(hits):
                    hit = hits[rank]
                    content = hit['content']
                    
                    text = content.get("chunk_text") or content.get("answer") or content.get("question") or ""
                    source = content.get("source_file", "")
                    score = hit.get('rrf_score', 0)

                    row[f"{method} Score"] = round(score, 4)
                    row[f"{method} Text"] = text[:32000]
                    row[f"{method} Source"] = source
                else:
                    row[f"{method} Score"] = None
                    row[f"{method} Text"] = ""
                    row[f"{method} Source"] = ""
            rows.append(row)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(rows).to_excel(output_path, index=False)
    print(f"Report saved to {output_path}")

def main():
    all_docs = load_documents(PATHS["docs"])
    if not all_docs:
        print("CRITICAL: No documents loaded. Exiting.")
        sys.exit(1)
    print(f"Total documents: {len(all_docs)}")

    print("\n--- Loading Models ---")
    try:
        bge_embedder = BGEHybridEmbedder(PATHS["bge"], device="cpu") 
        ft_embedder = FastTextEmbedder(PATHS["fasttext"])
    except Exception as e:
        print(f"Error loading models: {e}")
        sys.exit(1)

    print("\n--- Starting Indexing ---")
    
    ft_retriever = DenseRetriever(ft_embedder, name="FastText_Doc")
    ft_retriever.index(tqdm(all_docs, desc="Indexing FastText"), granularity="doc")

    bge_retriever = DenseRetriever(bge_embedder, name="BGE_Doc")
    bge_retriever.index(tqdm(all_docs, desc="Indexing BGE"), granularity="doc")

    bm25_retriever = BM25Retriever(name="BM25")
    bm25_retriever.index(tqdm(all_docs, desc="Indexing BM25"))

    print("\n--- Setting up Ensembles ---")
    pipelines = {
        "BGE_Only": EnsembleRetriever([bge_retriever]),
        "FastText_BM25": EnsembleRetriever([ft_retriever, bm25_retriever]),
        "All_Combined": EnsembleRetriever([bge_retriever, ft_retriever, bm25_retriever])
    }
    
    for p in pipelines.values():
        p.set_documents(all_docs)

    queries = load_queries(PATHS["queries"])[:QUERIES_LIMIT]
    if not queries:
        print("No queries found.")
        sys.exit(0)

    print(f"\nRunning comparison on {len(queries)} queries...")
    comparison_data = {}

    for q in tqdm(queries, desc="Processing Queries"):
        comparison_data[q] = {}
        for name, pipe in pipelines.items():
            comparison_data[q][name] = pipe.search(q, limit=TOP_K_RESULTS)

    output_file = os.path.join(PATHS["output_dir"], REPORT_FILENAME)
    save_report(comparison_data, output_file)

if __name__ == "__main__":
    main()