"""Sparse-dense retrieval fusion utilities."""

from __future__ import annotations

import csv
import json
import math
import os
import random
import re
import statistics
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


BEIR_URLS = {
    "scifact": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
    "nfcorpus": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
    "fiqa": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip",
    "quora": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/quora.zip",
    "arguana": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/arguana.zip",
}

DEFAULT_METHODS = ["bm25", "dense", "rrf", "zscore_fusion", "minmax_fusion", "validation_weighted_fusion"]
FUSION_METHODS = ["rrf", "zscore_fusion", "minmax_fusion", "validation_weighted_fusion"]
GATE_METHODS = ["validation_weighted_fusion", "rrf"]
WORD_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass
class RetrievalDataset:
    name: str
    corpus: Dict[str, str]
    queries: Dict[str, str]
    qrels: Dict[str, Dict[str, int]]
    split: str

    @property
    def query_ids(self) -> List[str]:
        return [qid for qid in self.queries if self.qrels.get(qid)]


def tokenize(text: str) -> List[str]:
    return WORD_RE.findall(text.lower())


def estimate_cell_count(
    datasets: Sequence[str],
    budgets: Sequence[int],
    methods: Sequence[str],
    seeds: Sequence[int],
) -> int:
    return len(datasets) * len(budgets) * len(methods) * len(seeds)


def evaluate_ranking(
    ranked_doc_ids: Sequence[str],
    qrels: Mapping[str, int],
    ndcg_k: int = 10,
    mrr_k: int = 10,
    recall_k: int = 100,
) -> Dict[str, float]:
    relevant = {doc_id: rel for doc_id, rel in qrels.items() if rel > 0}
    if not relevant:
        return {"ndcg@10": 0.0, "mrr@10": 0.0, "recall@100": 0.0}

    dcg = 0.0
    for rank, doc_id in enumerate(ranked_doc_ids[:ndcg_k], start=1):
        rel = relevant.get(doc_id, 0)
        if rel > 0:
            dcg += (2.0**rel - 1.0) / math.log2(rank + 1.0)
    ideal = sorted(relevant.values(), reverse=True)[:ndcg_k]
    idcg = sum((2.0**rel - 1.0) / math.log2(rank + 1.0) for rank, rel in enumerate(ideal, start=1))

    mrr = 0.0
    for rank, doc_id in enumerate(ranked_doc_ids[:mrr_k], start=1):
        if relevant.get(doc_id, 0) > 0:
            mrr = 1.0 / rank
            break

    recall_hits = sum(1 for doc_id in ranked_doc_ids[:recall_k] if relevant.get(doc_id, 0) > 0)
    return {
        "ndcg@10": float(dcg / idcg) if idcg else 0.0,
        "mrr@10": float(mrr),
        "recall@100": float(recall_hits / max(1, len(relevant))),
    }


def zscore_normalize(scores: Mapping[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    values = [float(value) for value in scores.values()]
    mean = statistics.fmean(values)
    stdev = statistics.pstdev(values)
    if stdev <= 1e-12:
        return {key: 0.0 for key in scores}
    return {key: (float(value) - mean) / stdev for key, value in scores.items()}


def minmax_normalize(scores: Mapping[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    values = [float(value) for value in scores.values()]
    minimum = min(values)
    maximum = max(values)
    span = maximum - minimum
    if span <= 1e-12:
        return {key: 0.0 for key in scores}
    return {key: (float(value) - minimum) / span for key, value in scores.items()}


def reciprocal_rank_fusion(
    bm25_ranking: Sequence[str],
    dense_ranking: Sequence[str],
    top_k: int,
    k: int = 60,
) -> List[str]:
    scores: Dict[str, float] = defaultdict(float)
    for ranking in [bm25_ranking, dense_ranking]:
        for rank, doc_id in enumerate(ranking[:top_k], start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]]


def weighted_zscore_fusion(
    bm25_scores: Mapping[str, float],
    dense_scores: Mapping[str, float],
    dense_weight: float,
    top_k: int,
) -> List[str]:
    candidates = sorted(set(bm25_scores) | set(dense_scores))
    bm25_norm = zscore_normalize({doc_id: float(bm25_scores.get(doc_id, 0.0)) for doc_id in candidates})
    dense_norm = zscore_normalize({doc_id: float(dense_scores.get(doc_id, 0.0)) for doc_id in candidates})
    sparse_weight = 1.0 - dense_weight
    fused = {
        doc_id: sparse_weight * bm25_norm.get(doc_id, 0.0) + dense_weight * dense_norm.get(doc_id, 0.0)
        for doc_id in candidates
    }
    return [doc_id for doc_id, _ in sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:top_k]]


def minmax_fusion(
    bm25_scores: Mapping[str, float],
    dense_scores: Mapping[str, float],
    dense_weight: float,
    top_k: int,
) -> List[str]:
    candidates = sorted(set(bm25_scores) | set(dense_scores))
    bm25_norm = minmax_normalize({doc_id: float(bm25_scores.get(doc_id, 0.0)) for doc_id in candidates})
    dense_norm = minmax_normalize({doc_id: float(dense_scores.get(doc_id, 0.0)) for doc_id in candidates})
    sparse_weight = 1.0 - dense_weight
    fused = {
        doc_id: sparse_weight * bm25_norm.get(doc_id, 0.0) + dense_weight * dense_norm.get(doc_id, 0.0)
        for doc_id in candidates
    }
    return [doc_id for doc_id, _ in sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:top_k]]


def split_validation_test_queries(
    query_ids: Sequence[str],
    seed: int,
    validation_fraction: float = 0.35,
    max_queries: int | None = None,
) -> Tuple[List[str], List[str]]:
    shuffled = list(query_ids)
    random.Random(seed).shuffle(shuffled)
    if max_queries is not None:
        shuffled = shuffled[: min(max_queries, len(shuffled))]
    if len(shuffled) < 4:
        raise ValueError("Need at least four qrels-backed queries for validation/test split")
    split = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * validation_fraction))))
    return shuffled[:split], shuffled[split:]


def _mean_metrics_for_qids(
    query_ids: Sequence[str],
    qrels: Mapping[str, Mapping[str, int]],
    rankings: Mapping[str, Sequence[str]],
) -> Dict[str, float]:
    metrics = [evaluate_ranking(rankings.get(qid, []), qrels[qid]) for qid in query_ids]
    if not metrics:
        return {"ndcg@10": 0.0, "mrr@10": 0.0, "recall@100": 0.0}
    return {
        metric: float(statistics.fmean(row[metric] for row in metrics))
        for metric in ["ndcg@10", "mrr@10", "recall@100"]
    }


def select_validation_weight(
    validation_qids: Sequence[str],
    qrels: Mapping[str, Mapping[str, int]],
    bm25_scores: Mapping[str, Mapping[str, float]],
    dense_scores: Mapping[str, Mapping[str, float]],
    weight_grid: Sequence[float],
) -> Dict[str, float]:
    best_weight = 0.5
    best_metric = -1.0
    best_mrr = -1.0
    for dense_weight in weight_grid:
        rankings = {
            qid: weighted_zscore_fusion(
                bm25_scores.get(qid, {}),
                dense_scores.get(qid, {}),
                dense_weight=float(dense_weight),
                top_k=100,
            )
            for qid in validation_qids
        }
        metrics = _mean_metrics_for_qids(validation_qids, qrels, rankings)
        score = metrics["ndcg@10"]
        if score > best_metric + 1e-12 or (abs(score - best_metric) <= 1e-12 and metrics["mrr@10"] > best_mrr):
            best_weight = float(dense_weight)
            best_metric = score
            best_mrr = metrics["mrr@10"]
    return {"dense_weight": best_weight, "validation_ndcg@10": best_metric, "validation_mrr@10": best_mrr}


def summarize_pilot_gate(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_cell: MutableMapping[Tuple[str, int, int], Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_cell[(str(row["dataset"]), int(row["budget"]), int(row["seed"]))][str(row["method"])] = row

    cell_summaries: List[Dict[str, Any]] = []
    method_reduces_avg_regret = {method: 0 for method in FUSION_METHODS}
    method_dataset: Dict[str, Dict[str, List[Dict[str, float]]]] = {
        method: defaultdict(list) for method in FUSION_METHODS
    }
    failures = []
    for (dataset, budget, seed), methods in sorted(by_cell.items()):
        if "bm25" not in methods or "dense" not in methods:
            failures.append({"dataset": dataset, "budget": budget, "seed": seed, "error": "missing single retriever"})
            continue
        bm25_ndcg = float(methods["bm25"]["ndcg@10"])
        dense_ndcg = float(methods["dense"]["ndcg@10"])
        bm25_mrr = float(methods["bm25"]["mrr@10"])
        dense_mrr = float(methods["dense"]["mrr@10"])
        best_single_ndcg = max(bm25_ndcg, dense_ndcg)
        average_single_ndcg = (bm25_ndcg + dense_ndcg) / 2.0
        worst_single_ndcg = min(bm25_ndcg, dense_ndcg)
        best_single_mrr = max(bm25_mrr, dense_mrr)
        average_single_mrr = (bm25_mrr + dense_mrr) / 2.0
        average_single_regret = best_single_ndcg - average_single_ndcg
        worst_single_regret = best_single_ndcg - worst_single_ndcg

        cell = {
            "dataset": dataset,
            "budget": budget,
            "seed": seed,
            "best_single_ndcg@10": best_single_ndcg,
            "average_single_ndcg@10": average_single_ndcg,
            "best_single_mrr@10": best_single_mrr,
            "average_single_mrr@10": average_single_mrr,
            "fusion": {},
        }
        for method in FUSION_METHODS:
            if method not in methods:
                continue
            ndcg = float(methods[method]["ndcg@10"])
            mrr = float(methods[method]["mrr@10"])
            method_regret = best_single_ndcg - ndcg
            reduces_avg = method_regret < average_single_regret - 1e-12
            if reduces_avg:
                method_reduces_avg_regret[method] += 1
            worst_reduction = (
                (worst_single_regret - method_regret) / worst_single_regret if worst_single_regret > 1e-12 else 0.0
            )
            item = {
                "ndcg_improvement_vs_average_single": ndcg - average_single_ndcg,
                "mrr_improvement_vs_average_single": mrr - average_single_mrr,
                "ndcg_regret_vs_best_single": method_regret,
                "worst_case_regret_reduction": worst_reduction,
                "reduces_average_single_regret": reduces_avg,
            }
            cell["fusion"][method] = item
            method_dataset[method][dataset].append(item)
        cell_summaries.append(cell)

    total_cells = len(by_cell)
    majority_threshold = total_cells / 2.0
    condition1_methods = [
        method for method, count in method_reduces_avg_regret.items() if count > majority_threshold
    ]

    method_dataset_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    condition2_methods = []
    for method in GATE_METHODS:
        dataset_summary: Dict[str, Dict[str, float]] = {}
        for dataset, values in method_dataset[method].items():
            dataset_summary[dataset] = {
                "mean_ndcg_improvement_vs_average_single": statistics.fmean(
                    value["ndcg_improvement_vs_average_single"] for value in values
                ),
                "mean_mrr_improvement_vs_average_single": statistics.fmean(
                    value["mrr_improvement_vs_average_single"] for value in values
                ),
                "mean_ndcg_regret_vs_best_single": statistics.fmean(
                    value["ndcg_regret_vs_best_single"] for value in values
                ),
                "mean_worst_case_regret_reduction": statistics.fmean(
                    value["worst_case_regret_reduction"] for value in values
                ),
            }
        method_dataset_summary[method] = dataset_summary
        ndcg_support = sum(1 for v in dataset_summary.values() if v["mean_ndcg_improvement_vs_average_single"] >= 0.015)
        mrr_support = sum(1 for v in dataset_summary.values() if v["mean_mrr_improvement_vs_average_single"] >= 0.015)
        worst_support = sum(1 for v in dataset_summary.values() if v["mean_worst_case_regret_reduction"] >= 0.20)
        if ndcg_support >= 2 or mrr_support >= 2 or worst_support >= 2:
            condition2_methods.append(method)

    candidate_methods = [method for method in GATE_METHODS if method in condition2_methods] or GATE_METHODS
    selected_method = min(
        candidate_methods,
        key=lambda method: max(
            [v["mean_ndcg_regret_vs_best_single"] for v in method_dataset_summary.get(method, {}).values()] or [999.0]
        ),
    )
    selected_dataset_summary = method_dataset_summary.get(selected_method, {})
    severe_degradation_datasets = [
        dataset
        for dataset, values in selected_dataset_summary.items()
        if values["mean_ndcg_regret_vs_best_single"] > 0.03
    ]

    criteria = {
        "fusion_reduces_average_regret_majority": bool(condition1_methods),
        "rrf_or_validation_weighted_supports_two_datasets": bool(condition2_methods),
        "no_systematic_severe_degradation": not severe_degradation_datasets,
    }
    return {
        "pass": all(criteria.values()) and not failures,
        "criteria": criteria,
        "selected_gate_method": selected_method,
        "condition1_methods": condition1_methods,
        "condition2_methods": condition2_methods,
        "method_reduces_average_regret_cells": method_reduces_avg_regret,
        "total_cells": total_cells,
        "method_cell_count": len(rows),
        "method_dataset_summary": method_dataset_summary,
        "severe_degradation_datasets": severe_degradation_datasets,
        "cell_summaries": cell_summaries,
        "failures": failures,
        "definitions": {
            "average_single_selection": "mean of BM25 and dense metrics in the same dataset/budget/seed cell",
            "best_single_retriever": "max of BM25 and dense nDCG@10 in the same dataset/budget/seed cell",
            "validation_weighted_fusion": "dense weight selected on validation queries only, evaluated on held-out test queries",
        },
    }


def ensure_beir_dataset(name: str, data_root: Path, allow_fetch: bool = True) -> Path:
    name = name.lower()
    if name not in BEIR_URLS:
        raise ValueError(f"Unknown BEIR dataset: {name}")
    dataset_dir = data_root / name
    if (dataset_dir / "corpus.jsonl").exists() and (dataset_dir / "queries.jsonl").exists():
        return dataset_dir
    if not allow_fetch:
        raise FileNotFoundError(f"{name} is not cached under {data_root}")
    data_root.mkdir(parents=True, exist_ok=True)
    zip_path = data_root / f"{name}.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(BEIR_URLS[name], zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_root)
    if not (dataset_dir / "corpus.jsonl").exists():
        raise FileNotFoundError(f"Downloaded {name} but corpus.jsonl was not found")
    return dataset_dir


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _qrels_path(dataset_dir: Path) -> Tuple[str, Path]:
    for split in ["test", "dev", "train"]:
        path = dataset_dir / "qrels" / f"{split}.tsv"
        if path.exists():
            return split, path
    raise FileNotFoundError(f"No qrels split found under {dataset_dir / 'qrels'}")


def load_beir_dataset(name: str, data_root: Path, allow_fetch: bool = True) -> RetrievalDataset:
    dataset_dir = ensure_beir_dataset(name, data_root=data_root, allow_fetch=allow_fetch)
    corpus = {}
    for row in _read_jsonl(dataset_dir / "corpus.jsonl"):
        doc_id = str(row.get("_id") or row.get("id"))
        corpus[doc_id] = " ".join(str(row.get(part, "")).strip() for part in ["title", "text"]).strip()

    queries = {}
    for row in _read_jsonl(dataset_dir / "queries.jsonl"):
        query_id = str(row.get("_id") or row.get("id"))
        queries[query_id] = str(row.get("text", "")).strip()

    split, qrels_file = _qrels_path(dataset_dir)
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    with qrels_file.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row or row[0] in {"query-id", "qid"} or len(row) < 3:
                continue
            qid, doc_id, score = str(row[0]), str(row[1]), row[2]
            try:
                rel = int(float(score))
            except ValueError:
                continue
            if qid in queries and doc_id in corpus and rel > 0:
                qrels[qid][doc_id] = rel
    return RetrievalDataset(name=name, corpus=corpus, queries=queries, qrels=dict(qrels), split=split)


class BM25Index:
    def __init__(self, doc_ids: Sequence[str], corpus: Mapping[str, str], k1: float = 1.5, b: float = 0.75):
        self.doc_ids = list(doc_ids)
        self.k1 = k1
        self.b = b
        self.doc_lengths: List[int] = []
        self.postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for idx, doc_id in enumerate(self.doc_ids):
            counts = Counter(tokenize(corpus[doc_id]))
            length = sum(counts.values())
            self.doc_lengths.append(length)
            for term, freq in counts.items():
                self.postings[term].append((idx, freq))
        self.avgdl = statistics.fmean(self.doc_lengths) if self.doc_lengths else 1.0
        n_docs = len(self.doc_ids)
        self.idf = {
            term: math.log(1.0 + (n_docs - len(postings) + 0.5) / (len(postings) + 0.5))
            for term, postings in self.postings.items()
        }

    def search_scores(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        import numpy as np

        scores = np.zeros(len(self.doc_ids), dtype="float32")
        for term in tokenize(query):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf[term]
            for idx, freq in postings:
                denom = freq + self.k1 * (1.0 - self.b + self.b * self.doc_lengths[idx] / max(self.avgdl, 1e-9))
                scores[idx] += idf * freq * (self.k1 + 1.0) / denom
        top_indices = scores.argsort()[::-1][:top_k]
        return [
            (self.doc_ids[int(idx)], float(scores[int(idx)]))
            for idx in top_indices
            if float(scores[int(idx)]) > 0.0
        ]


class TransformerEmbedder:
    def __init__(self, model_name: str, local_files_only: bool = True, device: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only).to(self.device)
        self.model.eval()

    def encode(self, texts: Sequence[str], batch_size: int = 64, max_length: int = 256):
        import numpy as np

        vectors = []
        with self.torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = list(texts[start : start + batch_size])
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                output = self.model(**encoded)
                token_embeddings = output.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
                pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                pooled = pooled / pooled.norm(p=2, dim=1, keepdim=True).clamp(min=1e-9)
                vectors.append(pooled.cpu().numpy())
        return np.vstack(vectors) if vectors else np.zeros((0, 1), dtype="float32")


def build_bm25_score_maps(dataset: RetrievalDataset, query_ids: Sequence[str], top_k: int) -> Tuple[Dict[str, Dict[str, float]], float]:
    index = BM25Index(list(dataset.corpus), dataset.corpus)
    started = time.perf_counter()
    score_maps = {
        qid: dict(index.search_scores(dataset.queries[qid], top_k=top_k))
        for qid in query_ids
    }
    latency_ms = (time.perf_counter() - started) * 1000.0 / max(1, len(query_ids))
    return score_maps, latency_ms


def build_dense_score_maps(
    dataset: RetrievalDataset,
    query_ids: Sequence[str],
    top_k: int,
    model_name: str,
    offline_models: bool,
    batch_size: int,
) -> Tuple[Dict[str, Dict[str, float]], float]:
    import numpy as np

    doc_ids = list(dataset.corpus)
    embedder = TransformerEmbedder(model_name, local_files_only=offline_models)
    started = time.perf_counter()
    doc_vectors = embedder.encode([dataset.corpus[doc_id] for doc_id in doc_ids], batch_size=batch_size)
    query_vectors = embedder.encode([dataset.queries[qid] for qid in query_ids], batch_size=batch_size)
    scores = query_vectors @ doc_vectors.T
    score_maps: Dict[str, Dict[str, float]] = {}
    for qidx, qid in enumerate(query_ids):
        top_indices = np.argsort(scores[qidx])[::-1][:top_k]
        score_maps[qid] = {doc_ids[int(idx)]: float(scores[qidx][int(idx)]) for idx in top_indices}
    latency_ms = (time.perf_counter() - started) * 1000.0 / max(1, len(query_ids))
    return score_maps, latency_ms


def _rank_scores(score_map: Mapping[str, float], top_k: int) -> List[str]:
    return [doc_id for doc_id, _ in sorted(score_map.items(), key=lambda item: (-item[1], item[0]))[:top_k]]


def _score_maps_for_budget(score_maps: Mapping[str, Mapping[str, float]], budget: int) -> Dict[str, Dict[str, float]]:
    return {qid: dict(sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:budget]) for qid, scores in score_maps.items()}


def evaluate_method(
    method: str,
    test_qids: Sequence[str],
    qrels: Mapping[str, Mapping[str, int]],
    bm25_scores: Mapping[str, Mapping[str, float]],
    dense_scores: Mapping[str, Mapping[str, float]],
    budget: int,
    dense_weight: float = 0.5,
) -> Dict[str, float]:
    rankings: Dict[str, List[str]] = {}
    for qid in test_qids:
        if method == "bm25":
            rankings[qid] = _rank_scores(bm25_scores.get(qid, {}), top_k=budget)
        elif method == "dense":
            rankings[qid] = _rank_scores(dense_scores.get(qid, {}), top_k=budget)
        elif method == "rrf":
            rankings[qid] = reciprocal_rank_fusion(
                _rank_scores(bm25_scores.get(qid, {}), top_k=budget),
                _rank_scores(dense_scores.get(qid, {}), top_k=budget),
                top_k=budget,
            )
        elif method == "zscore_fusion":
            rankings[qid] = weighted_zscore_fusion(
                bm25_scores.get(qid, {}),
                dense_scores.get(qid, {}),
                dense_weight=0.5,
                top_k=budget,
            )
        elif method == "minmax_fusion":
            rankings[qid] = minmax_fusion(
                bm25_scores.get(qid, {}),
                dense_scores.get(qid, {}),
                dense_weight=0.5,
                top_k=budget,
            )
        elif method == "validation_weighted_fusion":
            rankings[qid] = weighted_zscore_fusion(
                bm25_scores.get(qid, {}),
                dense_scores.get(qid, {}),
                dense_weight=dense_weight,
                top_k=budget,
            )
        else:
            raise ValueError(f"Unknown method: {method}")
    return _mean_metrics_for_qids(test_qids, qrels, rankings)


def run_pilot_matrix(
    data_root: Path,
    datasets: Sequence[str],
    budgets: Sequence[int],
    methods: Sequence[str],
    seeds: Sequence[int],
    max_queries: int,
    dense_model: str,
    offline_models: bool,
    allow_fetch: bool,
    batch_size: int,
    validation_fraction: float = 0.35,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    dataset_summaries: Dict[str, Any] = {}
    max_budget = max(max(budgets), 100)
    for dataset_name in datasets:
        dataset = load_beir_dataset(dataset_name, data_root=data_root, allow_fetch=allow_fetch)
        eligible = dataset.query_ids
        if len(eligible) < 4:
            failures.append({"dataset": dataset_name, "error": "fewer than four qrels-backed queries"})
            continue

        split_by_seed = {
            seed: split_validation_test_queries(
                eligible,
                seed=seed,
                validation_fraction=validation_fraction,
                max_queries=max_queries,
            )
            for seed in seeds
        }
        all_qids = sorted(set(qid for pair in split_by_seed.values() for group in pair for qid in group))
        bm25_all, bm25_latency = build_bm25_score_maps(dataset, all_qids, top_k=max_budget)
        dense_all, dense_latency = build_dense_score_maps(
            dataset,
            all_qids,
            top_k=max_budget,
            model_name=dense_model,
            offline_models=offline_models,
            batch_size=batch_size,
        )
        dataset_summaries[dataset_name] = {
            "split": dataset.split,
            "corpus_docs": len(dataset.corpus),
            "queries": len(dataset.queries),
            "qrels_queries": len(dataset.query_ids),
            "used_queries": len(all_qids),
            "bm25_latency_query_ms": bm25_latency,
            "dense_latency_query_ms": dense_latency,
        }

        for budget in budgets:
            bm25_budget = _score_maps_for_budget(bm25_all, budget)
            dense_budget = _score_maps_for_budget(dense_all, budget)
            for seed in seeds:
                validation_qids, test_qids = split_by_seed[seed]
                weight_info = select_validation_weight(
                    validation_qids=validation_qids,
                    qrels=dataset.qrels,
                    bm25_scores=bm25_budget,
                    dense_scores=dense_budget,
                    weight_grid=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                )
                for method in methods:
                    metrics = evaluate_method(
                        method=method,
                        test_qids=test_qids,
                        qrels=dataset.qrels,
                        bm25_scores=bm25_budget,
                        dense_scores=dense_budget,
                        budget=budget,
                        dense_weight=weight_info["dense_weight"],
                    )
                    latency_ms = {
                        "bm25": bm25_latency,
                        "dense": dense_latency,
                        "rrf": bm25_latency + dense_latency,
                        "zscore_fusion": bm25_latency + dense_latency,
                        "minmax_fusion": bm25_latency + dense_latency,
                        "validation_weighted_fusion": bm25_latency + dense_latency,
                    }[method]
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "split": dataset.split,
                            "budget": budget,
                            "seed": seed,
                            "method": method,
                            "validation_query_count": len(validation_qids),
                            "test_query_count": len(test_qids),
                            "dense_weight": weight_info["dense_weight"] if method == "validation_weighted_fusion" else "",
                            "validation_ndcg@10": weight_info["validation_ndcg@10"] if method == "validation_weighted_fusion" else "",
                            "validation_mrr@10": weight_info["validation_mrr@10"] if method == "validation_weighted_fusion" else "",
                            "latency_query_ms": latency_ms,
                            **metrics,
                        }
                    )

    gate = summarize_pilot_gate(rows)
    return {
        "status": "PASS" if gate["pass"] and not failures else "WEAK_SIGNAL_STOP",
        "rows": rows,
        "gate": gate,
        "failures": failures,
        "dataset_summaries": dataset_summaries,
        "config": {
            "datasets": list(datasets),
            "budgets": list(budgets),
            "methods": list(methods),
            "seeds": list(seeds),
            "max_queries": max_queries,
            "validation_fraction": validation_fraction,
            "dense_model": dense_model,
            "offline_models": offline_models,
            "expected_cell_count": estimate_cell_count(datasets, budgets, methods, seeds),
            "actual_cell_count": len(rows),
        },
    }


def set_offline_env(enabled: bool) -> None:
    value = "1" if enabled else "0"
    os.environ["TRANSFORMERS_OFFLINE"] = value
    os.environ["HF_DATASETS_OFFLINE"] = value
    os.environ["HF_HUB_OFFLINE"] = value
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def path_hygiene_issues(root: Path) -> List[str]:
    forbidden = [
        "/Users/" + "silver",
        "yaowen_sun" + "@hotmail.com",
        "sk" + "-",
        "open" + "ai",
        "cla" + "ude",
        "Evo" + "Scientist",
    ]
    issues = []
    for path in root.rglob("*"):
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if not path.is_file() or path.stat().st_size > 2_000_000:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in forbidden:
            if token in text:
                issues.append(f"{path}: contains {token}")
    return issues
