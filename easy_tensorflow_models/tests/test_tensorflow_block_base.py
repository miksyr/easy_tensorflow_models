from unittest import TestCase

from easy_tensorflow_models.tests.example_tensorflow_classes import BasicExampleBlock


class TestTensorflowBlock(TestCase):
    def __init__(self, methodName="runTest"):
        super().__init__(methodName=methodName)
        self.exampleBlock = BasicExampleBlock()

    def test_get_config(self):
        config = self.exampleBlock.get_config()
        expectedOutput = {"callReturnValue": self.exampleBlock.callReturnValue}
        for key, value in expectedOutput.items():
            self.assertEqual(config[key], value)
