from unittest import TestCase

from easy_tensorflow_models.tests.example_tensorflow_classes import BasicExampleBlock, ExampleTensorflowModel


class TestTensorflowModelBase(TestCase):
    def __init__(self, methodName="runTest"):
        super().__init__(methodName=methodName)
        self.exampleBlock = BasicExampleBlock()
        self.exampleModel = ExampleTensorflowModel(exampleBlock=self.exampleBlock)

    def test_get_model_blocks(self):
        blocks = self.exampleModel.get_model_blocks()
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0], self.exampleBlock)
