import unittest
import classification


class TestClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = classification.Classifier()

    def test_load_file(self):
        self.classifier.load_file('only_data.csv')
        print(self.classifier.origin_data[0])

    def test_separate_data(self):
        self.classifier.load_file('only_data.csv').separate_data()
        print(self.classifier.labels, self.classifier.features, sep="\n")

if __name__ == '__main__':
    unittest.main()