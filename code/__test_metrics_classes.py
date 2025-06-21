# -*- coding: utf-8 -*- noqa
"""
Created on Tue May  6 01:48:27 2025

@author: JoelT
"""
import torch
import unittest
# Replace with your module path
from metrics import BinaryClassMetrics, MultiLabelMetrics


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)

        # Binary test case
        self.binary_preds = torch.tensor(
            [1, 0, 1, 1, 0], dtype=torch.int32, device=self.device)
        self.binary_targets = torch.tensor(
            [1, 0, 0, 1, 0], dtype=torch.int32, device=self.device)

        # Multi-label test case: shape (batch_size, num_classes)
        self.multi_preds = torch.tensor([
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 1],
            [0, 0, 0]
        ], dtype=torch.int32, device=self.device)

        self.multi_targets = torch.tensor([
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ], dtype=torch.int32, device=self.device)

    def test_binary_metrics_run(self):
        """Test that all BinaryClassMetrics methods run and return floats."""
        metric = BinaryClassMetrics()
        metric.update(self.binary_preds, self.binary_targets)

        for name in dir(metric):
            if not name.startswith("_") and callable(getattr(metric, name)):
                method = getattr(metric, name)
                if hasattr(method, "__metric__"):  # tagged as a metric
                    result = method()
                    self.assertIsInstance(
                        result, float, f"{name} should return float")

    def test_multilabel_metrics_weighted_average(self):
        """
        Test that MultiLabelMetrics returns the correct weighted average
        for each metric based on positive support.
        """
        multilabel = MultiLabelMetrics(number_classes=3)
        multilabel.update(self.multi_preds, self.multi_targets)

        # Calculate expected values using BinaryClassMetrics for each class
        per_class_metrics = []
        supports = []

        for class_idx in range(self.multi_targets.shape[1]):
            bcm = BinaryClassMetrics()
            bcm.update(
                self.multi_preds[:, class_idx],
                self.multi_targets[:, class_idx]
            )
            per_class_metrics.append(bcm)
            supports.append(torch.sum(self.multi_targets[:, class_idx]).item())

        total_support = sum(supports) if sum(
            supports) > 0 else 1.0  # avoid div by zero

        for name in dir(multilabel):
            if not name.startswith("_") and callable(getattr(multilabel, name)):
                multilabel_method = getattr(multilabel, name)
                if hasattr(multilabel_method, "__metric__"):
                    # Weighted average of per-class metrics
                    expected = 0.0
                    for i, bcm in enumerate(per_class_metrics):
                        bcm_method = getattr(bcm, name, None)
                        if bcm_method is not None and callable(bcm_method):
                            expected += bcm_method() * supports[i]
                    expected /= total_support

                    actual = multilabel_method()
                    self.assertAlmostEqual(
                        expected, actual, places=5,
                        msg=(
                            f'{name}: Expected weighted average {expected},'
                            + f'got {actual}'
                        ),
                    )

    def test_batch_vs_all_at_once_consistency(self):
        """Test that updating batch-by-batch equals updating all at once."""
        full = MultiLabelMetrics(number_classes=3)
        full.update(self.multi_preds, self.multi_targets)

        batchwise = MultiLabelMetrics(number_classes=3)
        for pred_row, target_row in zip(self.multi_preds, self.multi_targets):
            batchwise.update(pred_row.unsqueeze(0), target_row.unsqueeze(0))

        for name in dir(full):
            if not name.startswith("_") and callable(getattr(full, name)):
                method_full = getattr(full, name)
                method_batch = getattr(batchwise, name)
                if hasattr(method_full, "__metric__"):
                    val_full = method_full()
                    val_batch = method_batch()
                    self.assertAlmostEqual(val_full, val_batch, places=5,
                                           msg=f"{name} differs: {val_full} vs {val_batch}")


if __name__ == "__main__":
    unittest.main()
