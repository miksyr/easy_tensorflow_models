from typing import Dict

import tensorflow as tf


class Metrics:
    def __init__(self):
        self.scalarMetrics = {}
        self.histogramMetrics = {}

    def add_scalar_metric(self, data: tf.Tensor, name: str) -> None:
        self.scalarMetrics.update({name: data})

    def add_histogram_metric(self, data: tf.Tensor, name: str) -> None:
        self.histogramMetrics.update({name: data})

    def get_histogram_metrics(self) -> Dict[str, tf.Tensor]:
        return self.histogramMetrics

    def get_scalar_metrics(self) -> Dict[str, tf.Tensor]:
        return self.scalarMetrics
