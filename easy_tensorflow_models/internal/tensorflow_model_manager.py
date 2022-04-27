from datetime import datetime
import json
from pathlib import Path
from typing import List, Union
from uuid import uuid4

import tensorflow as tf
from tqdm.auto import tqdm

from easy_tensorflow_models.store.batch_generator import BatchGenerator
from easy_tensorflow_models.internal.tensorflow_model_base import TensorflowModelBase


class TensorflowModelManager:
    def __init__(
        self,
        model: TensorflowModelBase,
        optimizer: tf.keras.optimizers.Optimizer,
        trainingBatchGenerator: BatchGenerator,
        evaluationBatchGenerator: BatchGenerator,
        projectPath: Union[Path, str],
    ):
        self.model = model
        self.optimizer = optimizer
        self.trainingBatchGenerator = trainingBatchGenerator
        self.evaluationBatchGenerator = evaluationBatchGenerator
        self.projectPath = projectPath

        self.dateMade = datetime.now().date()
        self.uuid = str(uuid4())[:6]
        self.globalStep = 0
        self.trainingSummaryWriter = None
        self.evaluationSummaryWriter = None
        tf.get_logger().setLevel(level="ERROR")

    @property
    def basePath(self) -> str:
        return f"{self.projectPath}/{self.dateMade.strftime('%Y%m%d')}/{self.model.modelName}/{self.uuid}"

    @property
    def modelSavePath(self) -> str:
        return f"{self.basePath}/model"

    @property
    def evaluationSavePath(self) -> str:
        return f"{self.basePath}/predictions"

    @property
    def tensorboardPath(self):
        return f"{self.basePath}/tensorboard"

    def initialise(self) -> None:
        Path(self.tensorboardPath).mkdir(parents=True, exist_ok=True)
        self.trainingSummaryWriter = tf.summary.create_file_writer(logdir=str(Path(self.tensorboardPath, "train")))
        self.evaluationSummaryWriter = tf.summary.create_file_writer(logdir=str(Path(self.tensorboardPath, "evaluate")))

    def save_model(self) -> None:
        directory = Path(self.modelSavePath, str(self.globalStep))
        directory.mkdir(parents=True, exist_ok=True)
        tf.saved_model.save(
            self.model,
            export_dir=str(directory),
            signatures={"predict": self.model.predict, "get_model_name": self.model.get_model_name},
        )
        for modelBlock in self.model.get_model_blocks():
            with open(f"{directory}/{modelBlock.blockName}.json", "w") as blockSaveFile:
                json.dump(modelBlock.get_config(), blockSaveFile)
        self.model.save_weights(filepath=f"{directory}/{self.model.modelName}.h5", save_format="h5")

    def train_step(self, withTracing: bool) -> tf.Tensor:
        inputs, targets = next(self.trainingBatchGenerator)
        if withTracing:
            tf.summary.trace_on(graph=True, profiler=True)
        with tf.GradientTape() as tape:
            loss = self.model.get_loss(targets=targets, training=True, **inputs)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        with self.trainingSummaryWriter.as_default():
            tf.summary.scalar(name="Loss", data=loss, step=self.globalStep)
            if withTracing:
                tf.summary.trace_export(
                    name=f"{self.model.modelName}{self.globalStep}",
                    step=self.globalStep,
                    profiler_outdir=str(Path(self.tensorboardPath, "trainGraph")),
                )
        return loss

    def evaluation_step(self, withTracing: bool) -> tf.Tensor:
        inputs, targets = next(self.evaluationBatchGenerator)
        if withTracing:
            tf.summary.trace_on(graph=True, profiler=True)
        loss = self.model.get_loss(targets=targets, training=False, **inputs)
        with self.evaluationSummaryWriter.as_default():
            tf.summary.scalar(name="Loss", data=loss, step=self.globalStep)
            metrics = self.model.produce_metrics(targets=targets, training=False, globalStep=self.globalStep, **inputs)
            for scalarName, scalarMetric in metrics.get_scalar_metrics().items():
                tf.summary.scalar(name=f"Evaluation/{scalarName}", data=scalarMetric, step=self.globalStep)
            for histogramName, histogramMetric in metrics.get_histogram_metrics().items():
                tf.summary.histogram(name=f"Evaluation/{histogramName}", data=histogramMetric, step=self.globalStep)
            if withTracing:
                tf.summary.trace_export(
                    name=f"{self.model.modelName}{self.globalStep}",
                    step=self.globalStep,
                    profiler_outdir=str(Path(self.tensorboardPath, "evaluateGraph")),
                )
        return loss

    def run(self, stepCount: int, evaluationSteps: List[int], tracingSteps: List[int]) -> tf.Tensor:
        self.initialise()
        currentLoss = 0
        for step in tqdm(range(stepCount)):
            if step in evaluationSteps:
                loss = self.evaluation_step(withTracing=step in tracingSteps)
                self.save_model()
                currentLoss = loss
            self.train_step(withTracing=step in tracingSteps)
            self.globalStep += 1
        self.save_model()
        return currentLoss
