import unittest
import classification


class TestClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = classification.Classifier()

    def test_load_data(self):
        self.classifier.load_data('only_data.csv')
        print(self.classifier.origin_data)

if __name__ == '__main__':
    unittest.main()