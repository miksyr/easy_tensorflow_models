from abc import ABC, abstractmethod

from easy_tensorflow_models.datamodel.data_batch import DataBatch


class BatchRetriever(ABC):
    @abstractmethod
    def get_batch(self, batchSize: int) -> DataBatch:
        pass
