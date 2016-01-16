import csv
import numpy as np


def load_data(filename):
    with open(filename, 'rt') as fin:
        cin = csv.reader(fin)
        data = [row for row in cin]

    return data


def separate_data(origin_data):
    label_name = np.array(origin_data[0][-1])
    feature_name = np.array(origin_data[0][0:-1])
    labels = np.array([i[-1] for i in origin_data[1:-1]])
    features = np.array([row[0:-1] for row in origin_data[1:-1]], dtype='float')

    return label_name, feature_name, labels, features
