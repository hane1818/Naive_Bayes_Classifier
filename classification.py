import csv


def load_data(filename):
    with open(filename, 'rt') as fin:
        cin = csv.reader(fin)
        data = [row for row in cin]

    return data
