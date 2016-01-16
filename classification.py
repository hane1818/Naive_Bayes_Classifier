import csv
import math


class Classifier:
    def __init__(self):
        self.origin_data = []
        self.features = []
        self.feature_name = []
        self.labels = []
        self.label_name = ""
        self.class_label = set()
        self.model = {}
        self.test_model = {}    # use for training more & more times

    def load_file(self, filename):
        with open(filename, 'rt') as fin:
            cin = csv.reader(fin)
            self.origin_data =[row for row in cin]
        return self

    def load_data(self, datasets):
        self.origin_data = np.array(datasets)

    def separate_data(self):
        try:
            self.label_name = self.origin_data[0][-1]
            self.feature_name = self.origin_data[0][0:-1]
            self.labels = [i[-1] for i in self.origin_data[1:-1]]
            self.features = [row[0:-1] for row in self.origin_data[1:-1]]
            for i, data in enumerate(self.features):
                for j, d in enumerate(data):
                    self.features[i][j] = float(d)
            self.class_label = set(self.labels)
        except IndexError:
            raise ValueError("no data in classifier, use load_data() to load a data set.")
        return self

    def train(self):
        # first add labels in model
        data_model = {}
        for i in self.class_label:
            data_model[i] = []

        # second separate data by label
        for i, data in enumerate(self.features):
            data_model[self.labels[i]].append(data)
        # calculate every label's probability
        for key in self.class_label:
            data_model[key] = (len(data_model[key])/len(self.class_label), data_model[key])

        # calculate every feature's probability
        def mean(numbers):
            return sum(numbers)/len(numbers)

        def stdev(numbers):
            avg = mean(numbers)
            variance = sum([pow(x-avg, 2) for x in numbers])/float(len(numbers)-1)
            return math.sqrt(variance)

        model = {}
        for label in self.class_label:
            model[label] = []
            for i in range(len(self.feature_name)):
                model[label].append([data[i] for data in data_model[label][1]])
                model[label][i] = (mean(model[label][i]), stdev(model[label][i]))
            model[label] = (data_model[label][0], model[label])

        if self.model == {}:
            self.model = model
        else:
            self.test_model = model

        return self
