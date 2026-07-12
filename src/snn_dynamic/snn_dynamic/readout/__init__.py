"""Feature collection and classifier evaluation."""

from snn_dynamic.readout.classifier import ClassificationReport, train_and_evaluate
from snn_dynamic.readout.features import FeatureBundle, collect_features

__all__ = [
    "ClassificationReport",
    "FeatureBundle",
    "collect_features",
    "train_and_evaluate",
]
