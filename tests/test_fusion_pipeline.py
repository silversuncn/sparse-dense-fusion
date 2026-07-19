import unittest

from fusion_pipeline import (
    DEFAULT_METHODS,
    evaluate_ranking,
    estimate_cell_count,
    minmax_fusion,
    reciprocal_rank_fusion,
    select_validation_weight,
    split_validation_test_queries,
    summarize_pilot_gate,
    weighted_zscore_fusion,
    zscore_normalize,
)


class FusionPipelineTest(unittest.TestCase):
    def test_zscore_normalize_handles_constant_scores(self):
        normalized = zscore_normalize({"a": 2.0, "b": 2.0})
        self.assertEqual(normalized, {"a": 0.0, "b": 0.0})

    def test_rrf_promotes_documents_supported_by_both_retrievers(self):
        fused = reciprocal_rank_fusion(["a", "b", "c"], ["c", "a", "d"], top_k=3, k=60)
        self.assertEqual(fused[0], "a")
        self.assertIn("c", fused[:3])

    def test_validation_split_is_deterministic_and_disjoint(self):
        qids = [f"q{i}" for i in range(20)]
        val_a, test_a = split_validation_test_queries(qids, seed=19, validation_fraction=0.4)
        val_b, test_b = split_validation_test_queries(qids, seed=19, validation_fraction=0.4)
        self.assertEqual(val_a, val_b)
        self.assertEqual(test_a, test_b)
        self.assertTrue(set(val_a).isdisjoint(test_a))
        self.assertGreater(len(val_a), 0)
        self.assertGreater(len(test_a), 0)

    def test_validation_weight_is_selected_without_test_queries(self):
        qrels = {
            "v1": {"d_bm25": 1},
            "v2": {"d_bm25": 1},
            "t1": {"d_dense": 1},
            "t2": {"d_dense": 1},
        }
        bm25_scores = {
            qid: {"d_bm25": 2.0, "d_dense": 0.0}
            for qid in qrels
        }
        dense_scores = {
            qid: {"d_bm25": 0.0, "d_dense": 2.0}
            for qid in qrels
        }
        selected = select_validation_weight(
            validation_qids=["v1", "v2"],
            qrels=qrels,
            bm25_scores=bm25_scores,
            dense_scores=dense_scores,
            weight_grid=[0.0, 0.5, 1.0],
        )
        self.assertEqual(selected["dense_weight"], 0.0)

    def test_weighted_zscore_fusion_and_metrics(self):
        ranking = weighted_zscore_fusion(
            {"a": 10.0, "b": 1.0},
            {"b": 5.0, "a": 0.0},
            dense_weight=0.5,
            top_k=2,
        )
        metrics = evaluate_ranking(ranking, {"a": 1, "b": 1})
        self.assertEqual(metrics["recall@100"], 1.0)

    def test_minmax_fusion_handles_disjoint_score_scales(self):
        ranking = minmax_fusion(
            {"shared": 2.0, "sparse_only": 1.0},
            {"shared": 10.0, "dense_only": 8.0},
            dense_weight=0.5,
            top_k=3,
        )
        self.assertEqual(ranking[0], "shared")
        self.assertIn("dense_only", ranking)
        self.assertIn("sparse_only", ranking)

    def test_formal_default_methods_include_minmax_fusion(self):
        self.assertIn("minmax_fusion", DEFAULT_METHODS)

    def test_pilot_gate_accepts_regret_reducing_fusion(self):
        rows = []
        for dataset in ["d1", "d2", "d3"]:
            for budget in [50, 100]:
                for seed in [19, 31, 47]:
                    rows.append(
                        {
                            "dataset": dataset,
                            "budget": budget,
                            "seed": seed,
                            "method": "bm25",
                            "ndcg@10": 0.40,
                            "mrr@10": 0.35,
                        }
                    )
                    rows.append(
                        {
                            "dataset": dataset,
                            "budget": budget,
                            "seed": seed,
                            "method": "dense",
                            "ndcg@10": 0.50,
                            "mrr@10": 0.45,
                        }
                    )
                    rows.append(
                        {
                            "dataset": dataset,
                            "budget": budget,
                            "seed": seed,
                            "method": "rrf",
                            "ndcg@10": 0.54,
                            "mrr@10": 0.49,
                        }
                    )
                    rows.append(
                        {
                            "dataset": dataset,
                            "budget": budget,
                            "seed": seed,
                            "method": "zscore_fusion",
                            "ndcg@10": 0.53,
                            "mrr@10": 0.48,
                        }
                    )
                    rows.append(
                        {
                            "dataset": dataset,
                            "budget": budget,
                            "seed": seed,
                            "method": "validation_weighted_fusion",
                            "ndcg@10": 0.55,
                            "mrr@10": 0.50,
                        }
                    )
        gate = summarize_pilot_gate(rows)
        self.assertTrue(gate["pass"])
        self.assertEqual(gate["total_cells"], 18)
        self.assertEqual(gate["method_cell_count"], 90)

    def test_estimate_cell_count_matches_plan(self):
        self.assertEqual(
            estimate_cell_count(
                datasets=["scifact", "nfcorpus", "fiqa"],
                budgets=[50, 100],
                methods=["bm25", "dense", "rrf", "zscore_fusion", "validation_weighted_fusion"],
                seeds=[19, 31, 47],
            ),
            90,
        )


if __name__ == "__main__":
    unittest.main()
