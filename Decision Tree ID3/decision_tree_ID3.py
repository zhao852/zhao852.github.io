# Edward Zhao
#
# Implements binary decision tree using ID3 (Iterative Dichotomiser 3) algorithm.
# Runs algorithm against a Titanic survivor data set, considering 7 attributes: Class of travel, sex, age, etc.
# First attribute "survived" is the class label.
# Implements 5-fold cross validation.

import numpy as np
import pandas as pd


def entropy(freqs):  # calculate entropy
    all_freq = sum(freqs)
    entr = 0
    for fq in freqs:
        prob = fq * 1.0 / all_freq
        if abs(prob) > 1e-8:
            entr += -prob * np.log2(prob)
    return entr


def infor_gain(before_split_freqs, after_split_freqs):  # calculate information gain
    gain = entropy(before_split_freqs)
    overall_size = sum(before_split_freqs)
    for freq in after_split_freqs:
        ratio = sum(freq) * 1.0 / overall_size
        gain -= ratio * entropy(freq)
    return gain


class Node(object):
    def __init__(self, l, r, attr, thresh, label1):
        self.left_subtree = l
        self.right_subtree = r
        self.attribute = attr
        self.threshold = thresh
        self.label = label1


def ID3(train_data, train_labels):
    name = list(train_labels.columns)[0]
    unique_list = list(set(train_labels[name]))
    label_list = list(train_labels[name])
    unique_list.sort()
    before_split_list = []
    attribute_list = list(train_data.columns)

    if train_data.empty:
        num = train_labels[name].mode()[0]
        return Node(None, None, None, None, num)
    for i in range(len(unique_list)):
        before_split_list.append(label_list.count(unique_list[i]))
    if len(attribute_list) <= 1:
        return Node(None, None, None, None, train_labels[name].mode()[0])
    if len(unique_list) <= 1:
        return Node(None, None, None, None, unique_list[0])

    info_gain = []
    thresh = []
    # calculate the infor-gain of every attribute
    # picking the threshold
    for att in train_data:
        info = []
        unique_att_list = list(set(train_data[att]))
        unique_att_list.sort()

        if len(unique_att_list) == 1:
            info_gain.append(0)
            thresh.append(-1)
            continue

        for i in range(len(unique_att_list)):
            left = train_labels[train_data[att] <= unique_att_list[i]]
            right = train_labels[train_data[att] > unique_att_list[i]]

            templ = []
            tempr = []
            for j in range(len(unique_list)):
                vall = list(left[name]).count(unique_list[j])
                valr = list(right[name]).count(unique_list[j])
                templ.append(vall)
                tempr.append(valr)

            if sum(tempr) == 0:
                continue

            # split the data using the threshold
            # calculate the infor_gain
            after_split_list = []
            after_split_list.append(templ)
            after_split_list.append(tempr)

            gain = infor_gain(before_split_list, after_split_list)
            info.append(gain)

        info_gain.append(max(info))
        ind = info.index(max(info))
        thresh.append(unique_att_list[ind])

    # picking the attribute that achieved the maximum infor-gain
    index = info_gain.index(max(info_gain))
    the_chosen_attribute = attribute_list[index]
    the_chosen_threshold = thresh[index]

    # split the data into two parts
    left_part_train_data = train_data[train_data[the_chosen_attribute] <= the_chosen_threshold]
    right_part_train_data = train_data[train_data[the_chosen_attribute] > the_chosen_threshold]

    left_part_train_label = train_labels[train_data[the_chosen_attribute] <= the_chosen_threshold]
    right_part_train_label = train_labels[train_data[the_chosen_attribute] > the_chosen_threshold]

    del (left_part_train_data[the_chosen_attribute])
    del (right_part_train_data[the_chosen_attribute])

    if len(list(set(left_part_train_label[name]))) == 0 or len(list(set(right_part_train_label[name]))) == 0:
        return Node(None, None, None, None, train_labels[name].mode()[0])

    lab = train_labels.iloc[:, 0].mode()[0]
    # build a node to hold the data
    current_node = Node(None, None, the_chosen_attribute, the_chosen_threshold, lab)

    # call ID3() for the left and right parts of the data
    left_subtree = ID3(left_part_train_data, left_part_train_label)
    right_subtree = ID3(right_part_train_data, right_part_train_label)
    current_node.left_subtree = left_subtree
    current_node.right_subtree = right_subtree

    return current_node


def print2DUtil(root, space):  # visualize tree
    if root is None:
        return
    space += COUNT[0]
    print2DUtil(root.right_subtree, space)
    print()
    for i in range(COUNT[0], space):
        print(end=" ")
    if root.attribute is None:
        print(root.label)
    else:
        print(root.attribute)
    print2DUtil(root.left_subtree, space)


def print2D(root):
    print2DUtil(root, 0)


if __name__ == "__main__":
    data = pd.read_csv("C:\\Users\\zhaoe\\OneDrive\\Documents\\Fall 2020\\cs 373\\CS37300 HW2\\titanic-train.data",
                       delimiter=',', index_col=None, engine='python')

    label = pd.read_csv("C:\\Users\\zhaoe\\OneDrive\\Documents\\Fall 2020\\cs 373\\CS37300 HW2\\titanic-train.label",
                       delimiter=',', index_col=None, engine='python')

    COUNT = [10]
    print2D(ID3(data, label))

    # predict on testing set & evaluate the testing accuracy
    for att in data:  # replacing missing values in the data with the mode
        data[att] = data[att].fillna(data[att].mode()[0])


    def k_fold(data, label):
        x = np.array_split(data, 5)
        y = np.array_split(label, 5)
        df1_list = []
        df3_list = []
        for i in range(len(x)):
            df = pd.DataFrame()
            df1 = pd.DataFrame()
            df2_valid = pd.DataFrame()
            df3 = pd.DataFrame()
            for j in range(len(x)):
                if i != j:
                    df = df.append(x[j])
                    df1 = df1.append(y[j])
                    df1_list.append(y[j])
                else:
                    df2_valid = df2_valid.append(x[j])
                    df3 = df3.append(y[j])
                    df3_list.append(y[j])

            root = ID3(df, df1)

            train_labels = []
            for index, row in df.iterrows():
                temp = root
                while temp.left_subtree is not None or temp.right_subtree is not None:
                    if row[temp.attribute] <= temp.threshold:
                        temp = temp.left_subtree
                    else:
                        temp = temp.right_subtree
                train_labels.append(temp.label)
            train_labels1 = []
            for index, row in df2_valid.iterrows():
                temp = root
                while temp.left_subtree is not None or temp.right_subtree is not None:
                    if row[temp.attribute] <= temp.threshold:
                        temp = temp.left_subtree
                    else:
                        temp = temp.right_subtree
                train_labels1.append(temp.label)

            df1_list = df1['survived'].tolist()
            df3_list = df3['survived'].tolist()

            acc = 0
            for k in range(len(train_labels)):
                if df1_list[k] == train_labels[k]:
                    acc += 1
            tra = (acc * 1.0) / len(train_labels)

            acc1 = 0
            for k in range(len(train_labels1)):
                if df3_list[k] == train_labels1[k]:
                    acc1 += 1
            val = (acc1 * 1.0) / len(train_labels1)

            print("fold " + str(i + 1) + ", train set accuracy= " + str(tra) + ", validation set accuracy= "
                  + str(val))

    k_fold(data, label)
