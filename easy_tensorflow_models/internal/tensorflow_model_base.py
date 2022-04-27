from abc import ABC, abstractmethod
import inspect
from typing import List, Tuple, Union

import tensorflow as tf
from tensorflow import keras

from easy_tensorflow_models.datamodel.metrics import Metrics
from easy_tensorflow_models.internal.blocks.tensorflow_block_base import TensorflowBlockBase


class TensorflowModelBase(keras.Model, ABC):
    def __init__(self, modelName: str = None, **kwargs):
        super().__init__(**kwargs)
        self.modelName = modelName or self.__class__.__name__

    def get_model_blocks(self) -> List[TensorflowBlockBase]:
        constructorParameters = inspect.signature(self.__class__.__init__).parameters.keys()
        return [v for k, v in self.__dict__.items() if k in constructorParameters and isinstance(v, TensorflowBlockBase)]

    @tf.function(input_signature=[])
    def get_model_name(self):
        return self.modelName

    @abstractmethod
    def get_loss(self, targets: Union[tf.Tensor, List[tf.Tensor]], training: bool, **kwargs) -> tf.Tensor:
        pass

    @abstractmethod
    def produce_metrics(
        self, targets: Union[tf.Tensor, List[tf.Tensor], Tuple[tf.Tensor]], training: bool, globalStep: int, **kwargs
    ) -> Metrics:
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        """Needs to have input signature for tf.function"""
        pass

    @abstractmethod
    def get_outputs(self, *args, **kwargs) -> Union[tf.Tensor, List[tf.Tensor], Tuple[tf.Tensor]]:
        pass
