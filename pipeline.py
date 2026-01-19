import os
import sys
import pickle
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
    "output_dir": "./comparison_results",
    "cache_dir": "./cache_indexes"
}
TOP_K_RESULTS = 10

def get_retriever(name: str, retriever_cls, documents, embedder = None, granularity: str = "doc"):
    os.makedirs(PATHS["cache_dir"], exist_ok=True)
    cache_path = os.path.join(PATHS["cache_dir"], f"{name}.pkl")
    
    if os.path.exists(cache_path):
        print(f"[{name}] Found cached index at {cache_path}. Loading...")
        
        try:
            with open(cache_path, 'rb') as f:
                retriever = pickle.load(f)
                if embedder is not None:
                    retriever.embedder = embedder
                    
                print(f"[{name}] Successfully loaded from cache")
                return retriever
        except Exception as e:
            print(f"[{name}] Failed to load cache ({e}). Proceeding to re-indexing")
            
    print(f"[{name}] Indexing documents...")
    if embedder is not None:
        retriever = retriever_cls(embedder, name=name)
    else:
        retriever = retriever_cls(name=name)
        
    retriever.index(tqdm(documents, desc=f"Indexing {name}"), granularity=granularity)   
    print(f"[{name}] Saving index to {cache_path}...")
    
    try:
        temp_embedder = None
        
        if hasattr(retriever, 'embedder'):
            temp_embedder = retriever.embedder
            retriever.embedder = None
            
        with open(cache_path, 'wb') as f:
            pickle.dump(retriever, f)
            
        if temp_embedder is not None:
            retriever.embedder = temp_embedder
        
        print(f"[{name}] Saved")
            
    except Exception as e:
        print(f"[{name}] Could not save cache: {e}")
        
    return retriever

    
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
                    
                    text = content.get("chunk_text") or (content.get("answer") + content.get("question")) or ""
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
    
    rv_retriever = get_retriever(
        name="RusVectores_Doc",
        retriever_cls=DenseRetriever,
        documents=all_docs,
        embedder=rv_embedder
    )
    
    ft_retriever = get_retriever(
        name="Fasttex_Doc",
        retriever_cls=DenseRetriever,
        documents=all_docs,
        embedder=ft_embedder
    )

    bge_retriever = get_retriever(
        name="BGE_Doc",
        retriever_cls=DenseRetriever,
        documents=all_docs,
        embedder=bge_embedder
    )

    bm25_retriever = get_retriever(
        name="BM25_Doc", 
        retriever_cls=BM25Retriever,
        documents=all_docs,
    )

    print("\n--- Setting up Ensembles ---")
    pipelines = {
        "BGE_Only": EnsembleRetriever([bge_retriever]),
        "FastText_FB_Only": EnsembleRetriever([ft_retriever]),
        "RusVectores_Only": EnsembleRetriever([rv_retriever]),
        "BM25_Only": EnsembleRetriever([bm25_retriever])
    }
    
    for p in pipelines.values():
        p.set_documents(all_docs)
        
    i=0
    while True:
        i+=1
        query = str(input("Enter your query\n")).strip()
        
        comparison_data = {}

        comparison_data[query] = {}
        for name, pipe in pipelines.items():
            comparison_data[query][name] = pipe.search(query, limit=TOP_K_RESULTS)
            report_filename = f"{name}_{i}.xlsx"
            output_file = os.path.join(PATHS["output_dir"], report_filename)
            save_report(comparison_data, output_file)
        print(f"Query index: {i}")

if __name__ == "__main__":
    main()