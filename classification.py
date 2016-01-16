import csv
import math
import operator
import functools


class Classifier:
    def __init__(self):
        self.origin_data = []
        self.features = []
        self.feature_name = []
        self.labels = []
        self.label_name = []    # maybe it
        self.class_label = set()
        self.model = {}
        self.test_model = {}    # use for training more & more times
        self.accuracy = 0

    def load_file(self, filename):
        with open(filename, 'rt') as fin:
            cin = csv.reader(fin)
            self.origin_data = [row for row in cin]
        return self

    def load_data(self, datasets):
        self.origin_data = datasets

    def separate_data(self, label_num=1):
        try:
            self.label_name = self.origin_data[0][-label_num:]
            self.feature_name = self.origin_data[0][0:-label_num]
            self.labels = [i[-label_num] for i in self.origin_data[1:]]
            self.features = [row[0:-label_num] for row in self.origin_data[1:]]
            for i, data in enumerate(self.features):
                for j, d in enumerate(data):
                    self.features[i][j] = float(d)
            self.class_label = set(self.labels)
        except IndexError:
            raise ValueError("no data in classifier, use load_data() to load a data set.")
        return self

    def train(self):
        if not self.features:
            raise ValueError("no separated data in classifier, use load_data() & separate_data() first")
        # first add labels in model
        data_model = {}
        for i in self.class_label:
            data_model[i] = []

        # second separate data by label
        for i, data in enumerate(self.features):
            data_model[self.labels[i]].append(data)
        # calculate every label's probability
        for key in self.class_label:
            data_model[key] = (len(data_model[key])/len(self.features), data_model[key])

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

        if not self.model:
            self.model = model
        else:
            self.test_model = model

        return self

    def test(self, dataset):
        if not self.model:
            raise ValueError("need to train data before test")
        elif type(dataset) is not list or \
                type(dataset[0]) is not list and len(dataset) != len(self.feature_name + self.label_name) or \
                type(dataset[0]) is list and len(dataset[0]) != len(self.feature_name + self.label_name) :
            raise ValueError("invalid test data")

        feature = []
        actual = []
        label_num = len(self.label_name)
        if type(dataset) is list and type(dataset[0]) is not list:
            feature.append(dataset[0:-label_num])
            actual.append(dataset[-label_num:])
        else:
            for data in dataset:
                feature.append(data[0:-1])
                actual.append(data[-1])

        result = self.fit(feature)
        correct = 0
        for i, x in enumerate(actual):
            if result[i] == x:
                correct += 1
        accuracy = correct/len(actual)
        if accuracy > self.accuracy:
            self.accuracy = accuracy
            self.model = self.test_model if self.test_model else self.model
        self.test_model = {}

        return self

    def fit(self, dataset):
        if not self.model:
            raise ValueError("need to train data before test")
        elif type(dataset) is not list or \
                type(dataset[0]) is not list and len(dataset) != len(self.feature_name) or \
                type(dataset[0]) is list and len(dataset[0]) != len(self.feature_name):
            raise ValueError("invalid test data")
        feature = []
        if type(dataset) is list and type(dataset[0]) is not list:
            feature.append(dataset)
        else:
            feature = dataset

        model = self.model if not self.test_model else self.test_model
        predict_result = {}
        for label in self.class_label:
            predict_result[label] = []
            for data in feature:
                prob = []
                for i, x in enumerate(data):
                    prob.append(calc_probability(x, model[label][1][i][0], model[label][1][i][1]))
                prob = functools.reduce(operator.mul, prob, 1) * model[label][0]
                predict_result[label].append(prob)

        result_list = []
        for i in range(len(feature)):
            result_list.append(max([predict_result[label][i] for label in self.class_label]))

        for key, data in predict_result.items():
            for i, prob in enumerate(data):
                if prob == result_list[i]:
                    result_list[i] = key

        return result_list


def calc_probability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean, 2)/(2*math.pow(stdev, 2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent


def main():
    pass
if __name__ == '__main__':
    main()
