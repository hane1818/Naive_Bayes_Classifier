import unittest
import classification


class TestClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = classification.Classifier('only_data.csv')