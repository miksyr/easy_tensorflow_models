from easy_tensorflow_models.store.batch_retriever import BatchRetriever
from easy_tensorflow_models.store.batch_processor_base import BatchProcessorBase


class BatchGenerator:
    def __init__(self, batchRetriever: BatchRetriever, batchSize: int, batchProcessor: BatchProcessorBase):
        self.batchRetriever = batchRetriever
        self.batchSize = batchSize
        self.batchProcessor = batchProcessor

    def __next__(self):
        batch = self.batchRetriever.get_batch(batchSize=self.batchSize)
        return self.batchProcessor.process_batch(batch=batch)
