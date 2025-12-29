import os
import sys
import pandas as pd
from tqdm import tqdm
from typing import Dict

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from database import JSONLDataset, XLSXDataset
from retrieval import DenseRetriever, BM25Retriever, EnsembleRetriever
from embedders import BGEHybridEmbedder, FastTextEmbedder, RusVectoresEmbedder

PATHS = {
    "rusvectores": "/home/mikhailovnk/00_Models/Embedders/rusvectores/model.model",
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

def save_report(results_map: Dict, output_path: str):
    if not results_map:
        print("No results to save.")
        return

    rows = []

    try:
        first_q = next(iter(results_map))
        methods = list(results_map[first_q].keys())
    except StopIteration:
        print("Results map is empty.")
        return

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
    print("\n--- Loading Documents ---")
    all_docs = []
    
    for path in PATHS["docs"]:
        if not os.path.exists(path):
            print(f"Warning: Path not found {path}")
            continue
            
        print(f"Reading {os.path.basename(path)}...")
        ds = JSONLDataset(path)
        
        for i in tqdm(range(len(ds)), desc=f"Parsing {os.path.basename(path)}"):
            rec = ds[i]
            rec['source_file'] = os.path.basename(path)
            all_docs.append(rec)

    if not all_docs:
        print("CRITICAL: No documents loaded. Exiting.")
        sys.exit(1)
    print(f"Total documents loaded: {len(all_docs)}")

    print("\n--- Loading Models ---")
    try:
        bge_embedder = BGEHybridEmbedder(PATHS["bge"], device="cpu") 
        ft_embedder = FastTextEmbedder(PATHS["fasttext"])
        rv_embedder = RusVectoresEmbedder(PATHS["rusvectores"]) 
    except Exception as e:
        print(f"Error loading models: {e}")
        sys.exit(1)

    print("\n--- Starting Indexing ---")
    
    rv_retriever = DenseRetriever(rv_embedder, name="RusVectores_Doc")
    rv_retriever.index(tqdm(all_docs, desc="Indexing RusVectores"), granularity="doc")
    
    ft_retriever = DenseRetriever(ft_embedder, name="FastText_Doc")
    ft_retriever.index(tqdm(all_docs, desc="Indexing FastText"), granularity="doc")

    bge_retriever = DenseRetriever(bge_embedder, name="BGE_Doc")
    bge_retriever.index(tqdm(all_docs, desc="Indexing BGE"), granularity="doc")

    bm25_retriever = BM25Retriever(name="BM25")
    bm25_retriever.index(tqdm(all_docs, desc="Indexing BM25"))

    print("\n--- Setting up Ensembles ---")
    pipelines = {
        "BGE_Only": EnsembleRetriever([bge_retriever]),
        "FastText_FB_Only": EnsembleRetriever([ft_retriever]),
        "RusVectores_Only": EnsembleRetriever([rv_retriever]),
        "All_Ensemble": EnsembleRetriever([bge_retriever, rv_retriever, bm25_retriever])
    }
    
    for p in pipelines.values():
        p.set_documents(all_docs)

    print("\n--- Loading Queries ---")
    if not os.path.exists(PATHS["queries"]):
        print(f"CRITICAL: Queries file not found: {PATHS['queries']}")
        sys.exit(1)

    queries_ds = XLSXDataset(PATHS["queries"])
    
    target_col = next((col for col in queries_ds.columns if 'Вопрос' in str(col)), None)
    
    if not target_col:
        print("CRITICAL: Column 'Вопрос' not found in Excel.")
        sys.exit(1)
    
    total_queries = min(len(queries_ds), QUERIES_LIMIT)
    print(f"\nRunning comparison on {total_queries} queries...")
    
    comparison_data = {}

    for i in tqdm(range(total_queries), desc="Processing Queries"):
        row = queries_ds[i]
        query_text = str(row.get(target_col, "")).strip()

        if not query_text:
            continue

        comparison_data[query_text] = {}
        for name, pipe in pipelines.items():
            comparison_data[query_text][name] = pipe.search(query_text, limit=TOP_K_RESULTS)

    output_file = os.path.join(PATHS["output_dir"], REPORT_FILENAME)
    save_report(comparison_data, output_file)

if __name__ == "__main__":
    main()