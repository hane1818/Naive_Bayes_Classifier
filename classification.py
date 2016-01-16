import csv
import numpy as np


class Classifier:
    def __init__(self):
        self.origin_data = []
        self.features = []
        self.feature_name = []
        self.labels = []
        self.label_name = ""

    def load_file(self, filename):
        with open(filename, 'rt') as fin:
            cin = csv.reader(fin)
            self.origin_data = np.array([row for row in cin])
        return self

    def load_data(self, datasets):
        self.origin_data = np.array(datasets)

    def separate_data(self):
        try:
            self.label_name = np.array(self.origin_data[0][-1])
            self.feature_name = np.array(self.origin_data[0][0:-1])
            self.labels = np.array([i[-1] for i in self.origin_data[1:-1]])
            self.features = np.array([row[0:-1] for row in self.origin_data[1:-1]], dtype='float')
        except IndexError:
            raise ValueError("no data in classifier, use load_data() to load a data set.")
        return self

    def train(self):
        pass
