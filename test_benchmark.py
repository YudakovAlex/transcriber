import tempfile
import unittest
from pathlib import Path

from experiments.modal_benchmark import (matrix, peak_concurrency, reconcile,
                                        reserve_budget, word_errors)


class BenchmarkTests(unittest.TestCase):
    def test_matrix_keeps_parallelism_comparable(self):
        cells = matrix(Path("example.mp4"))
        self.assertEqual(len(cells), 30)
        self.assertEqual({c["workers"] for c in cells if c["backend"] == "modal"}, {1, 2, 4})
        self.assertEqual(sum(c["state"] == "cold" for c in cells), 15)

    def test_word_error_counts(self):
        self.assertEqual(word_errors("Hello, WORLD!", "hello world")["wer"], 0)
        self.assertEqual(word_errors("one two three", "one three")["deletions"], 1)
        self.assertEqual(word_errors("one two", "one extra two")["insertions"], 1)
        self.assertEqual(word_errors("one two", "one three")["substitutions"], 1)
        self.assertIsNone(word_errors("", "hallucination")["wer"])

    def test_observed_concurrency_counts_overlaps(self):
        results = [{"execution": {"started_at": s, "finished_at": e}}
                   for s, e in ((0, 3), (1, 2), (3, 5))]
        self.assertEqual(peak_concurrency(results), 2)

    def test_reservations_survive_until_actual_cost_reconciliation(self):
        root = Path(__file__).resolve().parent / ".tmp_test"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            output = Path(directory)
            reserve_budget(output, "first", 1, 0.75)
            with self.assertRaisesRegex(ValueError, "Budget gate"):
                reserve_budget(output, "second", 1, 0.5)
            reconcile(output, "first", 0.1)
            reserve_budget(output, "second", 1, 0.5)
            with self.assertRaises(ValueError):
                reserve_budget(output, "bad", 1, float("nan"))


if __name__ == "__main__":
    unittest.main()
