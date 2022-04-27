from typing import Union
from pathlib import Path

import tensorflow as tf


class TaggerBase:
    def __init__(self, modelDirectory: Union[str, Path], useGPU: bool = True, threads: int = None):
        self.modelDirectory = modelDirectory
        self.useGPU = useGPU
        if not self.useGPU:
            tf.config.experimental.set_visible_devices([], "GPU")
        self.threads = threads
        if threads is not None:
            tf.config.threading.set_inter_op_parallelism_threads(self.threads)
            tf.config.threading.set_intra_op_parallelism_threads(self.threads)
        self.model = tf.saved_model.load(export_dir=modelDirectory)

    @property
    def modelName(self) -> str:
        return self.model.get_model_name().numpy().decode()

    @property
    def modelIdentifier(self) -> str:
        return str(Path(self.modelDirectory).stem)
