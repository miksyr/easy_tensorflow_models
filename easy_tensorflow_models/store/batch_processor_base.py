from abc import ABC, abstractmethod
from typing import Any

from easy_tensorflow_models.datamodel.data_batch import DataBatch


class BatchProcessorBase(ABC):
    @abstractmethod
    def process_batch(self, batch: DataBatch) -> Any:
        pass
