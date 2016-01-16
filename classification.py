import csv
import numpy as np


class Classifier:
    def __init__(self, training_file):
        self.training_file = training_file
        self.origin_data = []
        self.features = []
        self.feature_name = []
        self.labels = []
        self.label_name = ""

    def load_data(self):
        with open(self.training_file, 'rt') as fin:
            cin = csv.reader(fin)
            self.origin_data = [row for row in cin]

    def separate_data(self):
        self.label_name = np.array(self.origin_data[0][-1])
        self.feature_name = np.array(self.origin_data[0][0:-1])
        self.labels = np.array([i[-1] for i in self.origin_data[1:-1]])
        self.features = np.array([row[0:-1] for row in self.origin_data[1:-1]], dtype='float')
