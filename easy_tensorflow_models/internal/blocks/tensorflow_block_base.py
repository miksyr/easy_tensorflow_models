from abc import ABC, abstractmethod
import inspect
from typing import Any, Dict

from tensorflow.python.keras import layers


class TensorflowBlockBase(layers.Layer, ABC):
    def __init__(self, blockName: str = None, **kwargs):
        super().__init__(self, **kwargs)
        self.blockName = blockName or self.__class__.__name__

    def get_config(self) -> Dict[str, Any]:
        constructorParameters = inspect.signature(self.__class__.__init__).parameters.keys()
        return {name: value for name, value in self.__dict__.items() if name in constructorParameters}

    @abstractmethod
    def get_outputs(self, inputs, *args, **kwargs):
        pass
