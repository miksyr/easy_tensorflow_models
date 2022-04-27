from abc import ABC
from typing import Any, List


class DataBatch(ABC):
    def __init__(self, data: List[Any]):
        self.data = data

    def __iter__(self):
        for value in self.data:
            yield value

    def __len__(self) -> int:
        return len(self.data)
