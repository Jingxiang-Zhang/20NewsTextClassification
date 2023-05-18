import numpy as np
import os
import copy
import scipy

base_path = r"20news/matlab"

train_data_path = os.path.join(base_path, "train.data")
train_label_path = os.path.join(base_path, "train.label")
train_map_path = os.path.join(base_path, "train.map")

test_data_path = os.path.join(base_path, "test.data")
test_label_path = os.path.join(base_path, "test.label")
test_map_path = os.path.join(base_path, "test.map")

vocabulary_path = os.path.join(base_path, "vocabulary.txt")


def read_map(path):
    # read .map file
    label_map = dict()
    with open(path, "r") as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip().split()
            label_map[int(line[1])] = line[0]
    return label_map


def read_words(path):
    # read vocabulary.txt file
    words = list()
    with open(path, "r") as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            words.append(line)
    return words


def read_data(data_path, label_path):
    # read .data file
    # read the label
    label = list()
    with open(label_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            label.append(int(line))
    label = np.asarray(label)

    data = dict()
    # each data is a list, each item in the list is a pair, the first is the location,
    # the second is value.
    with open(data_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip().split()
            index = int(line[0])
            w = int(line[1])
            value = int(line[2])
            if data.get(index):  # data exist
                data[index].append((w, value))
            else:  # not exist
                data[index] = [(w, value)]
    data_ret = list()
    for key, value in data.items():
        data_ret.append(value)
    return data_ret, label


def show_data(data_item, words_list, label, map_label):
    print("data label: {}".format(map_label[label]))
    data_item = copy.deepcopy(data_item)
    for i, item in enumerate(data_item):
        print("{}:{}".format(words_list[item[0] - 1], item[1]), end=" ")


def to_sparse_matrix(data, matrix_length):
    data_length = len(data)
    sparse = np.zeros(shape=(data_length, matrix_length), dtype=np.float32)
    for i, item in enumerate(data):
        for w, v in item:
            sparse[i][w - 1] = v
    return sparse


def to_scipy_sparse_matrix(data, length):
    data_ = list()
    row_ind = list()
    col_ind = list()
    for i, item in enumerate(data):
        for w, v in item:
            data_.append(v)
            row_ind.append(i)
            col_ind.append(w)
    csc_matrix = scipy.sparse.csc_matrix((data_, (row_ind, col_ind)), shape=(len(data), length))
    return csc_matrix


def spearate(data, label):
    data = np.asarray(data,dtype=np.object)
    label = np.asarray(label)
    # find the number of each class
    index = np.random.permutation(len(label))
    number = len(label)
    separate_point = int(number * 0.2)
    learning_index = index[separate_point:]
    val_index = index[:separate_point]
    
    train_data = data[learning_index]
    train_label = label[learning_index]
    val_data = data[val_index]
    val_label = label[val_index]
    return train_data, train_label, val_data, val_label
