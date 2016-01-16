import csv
import math
import operator
import functools
import random


class Classifier:
    def __init__(self):
        self.origin_data = []
        self.features = []
        self.feature_num = 0
        self.labels = []
        self.label_num = 0    # maybe it
        self.class_label = set()
        self.model = {}
        self.test_model = {}    # use for training more & more times
        self.train_accuracy = 0
        self.test_accuracy = 0

    def load_data(self, datasets):
        self.origin_data = datasets
        return self

    def separate_data(self, label_num=1):
        try:
            self.label_num = label_num
            self.feature_num = len(self.origin_data[0][0:-label_num])
            self.labels = [i[-label_num] for i in self.origin_data[1:]]
            self.features = [row[0:-label_num] for row in self.origin_data[1:]]
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
            for i in range(self.feature_num):
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
                type(dataset[0]) is not list and len(dataset) != self.feature_num + self.label_num or \
                type(dataset[0]) is list and len(dataset[0]) != self.feature_num + self.label_num:
            raise ValueError("invalid test data")

        feature = []
        actual = []
        label_num = self.label_num
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
        self.test_accuracy = accuracy
        if accuracy > self.train_accuracy:
            self.train_accuracy = accuracy
            self.model = self.test_model if self.test_model else self.model
        self.test_model = {}

        return accuracy

    def fit(self, dataset):
        if not self.model:
            raise ValueError("need to train data before test")
        elif type(dataset) is not list or \
                type(dataset[0]) is not list and len(dataset) != self.feature_num or \
                type(dataset[0]) is list and len(dataset[0]) != self.feature_num:
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


def bootstrap(dataset, split_ratio, times):
    train_size = int(len(dataset) * split_ratio)
    test_size = len(dataset) - train_size
    for i in range(times):
        train_set = []
        test_set = []
        while len(train_set) < train_size:
            index = random.randrange(len(dataset))
            train_set.append(dataset[index])
        while len(test_set) < test_size:
            index = random.randrange(len(dataset))
            test_set.append(dataset[index])
        yield train_set, test_set

def k_fold(dataset, fold):
    test_size = len(dataset) // fold
    start = 0
    end = test_size
    for i in range(fold):
        yield dataset[0:start]+dataset[end:-1], dataset[start:end]
        start += test_size
        end += test_size

def main():
    nb = Classifier()
    # load data
    with open('train_data.csv', 'r') as fin:
        cin = csv.reader(fin)
        train_data = [row for row in cin]

    with open('test_data.csv', 'r') as fin:
        cin = csv.reader(fin)
        test_data = [row for row in cin]

    # preprocessing
    title = train_data.pop(0)
    for i, data in enumerate(train_data):
        for j, d in enumerate(data):
            try:
                train_data[i][j] = float(d)
            except ValueError:
                pass
    for i, data in enumerate(test_data):
        for j, d in enumerate(data):
            try:
                test_data[i][j] = float(d)
            except ValueError:
                pass

    # training & testing
    print("{:30} : {}".format("Number of training data sets", len(train_data)),
          "{:30} : {}".format("Number of testing data sets", len(test_data)), sep="\n")
    print("\n==========================================\n")
    nb.load_data(train_data).separate_data().train()
    nb.test(test_data)
    print("Use training data sets training: \n",
          "\tTraining accuracy\t= {:.2f}%\n\tTesting accuracy\t= {:.2f}%".format(nb.train_accuracy*100, nb.test_accuracy*100))
    print("\n==========================================\n")
    for trainset, testset in bootstrap(train_data, 0.673, len(train_data)):
        nb.load_data(trainset).separate_data().train()
        nb.test(testset)
    nb.test(test_data)
    print("After bootstrap evaluation: \n",
          "\tTraining accuracy\t= {:.2f}%\n\tTesting accuracy\t= {:.2f}%".format(nb.train_accuracy*100, nb.test_accuracy*100))
    print("\n==========================================\n")
    for trainset, testset in k_fold(train_data, len(train_data)):
        nb.load_data(trainset).separate_data().train()
        nb.test(testset)
    nb.test(test_data)
    print("After k-fold evaluation: \n",
          "\tTraining accuracy\t= {:.2f}%\n\tTesting accuracy\t= {:.2f}%".format(nb.train_accuracy*100, nb.test_accuracy*100))
    print("\n==========================================\n")
    nb.load_data(test_data).separate_data().train()
    nb.test(test_data)
    print("After training via testing data sets: \n"
          "\tTraining accuracy\t= {:.2f}%\n\tTesting accuracy\t= {:.2f}%".format(nb.train_accuracy*100, nb.test_accuracy*100))
    print("\n==========================================\n")
    nb.test(train_data)
    print("Use training data set for test : \n"
          "\tTraining accuracy\t= {:.2f}%\n\tTesting accuracy\t= {:.2f}%".format(nb.train_accuracy*100, nb.test_accuracy*100))

if __name__ == '__main__':
    main()
