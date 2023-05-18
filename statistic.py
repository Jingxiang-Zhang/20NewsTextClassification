import pandas as pd
import os
import numpy as np


def compare_prediction(prediction, label, class_number=20):
    ratio_list = list()
    for i in range(1, class_number + 1):
        i_label_index = label==i
        class_number = np.sum(i_label_index)
        correct_prediction = np.sum(prediction[i_label_index] == i)
        ratio = correct_prediction / class_number
        ratio_list.append(ratio)
    return np.asarray(ratio_list)

def compare_prediction(prediction, label, class_number=20):
    ratio_list = list()
    for i in range(1, class_number + 1):
        i_label_index = label==i
        class_number = np.sum(i_label_index)
        correct_prediction = np.sum(prediction[i_label_index] == i)
        ratio = correct_prediction / class_number
        ratio_list.append(ratio)
    return np.asarray(ratio_list)

def save_to_file(method_name, ratio_list, accuracy, class_list, filename="save.xlsx"):
    if os.path.exists(filename):
        df = pd.read_excel(filename)
    else:
        df = pd.DataFrame(columns=["method", *class_list, "sum"])
    df.loc[len(df)] = [method_name, *ratio_list, accuracy]
    df.to_excel(filename, index=False)
