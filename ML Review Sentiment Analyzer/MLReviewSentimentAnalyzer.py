# Program that reads a review and determines whether it's positive or negative.
# Edward Zhao

import re
from collections import Counter


def read_and_divide_by_sentiment(file_name):
    try:
        file1 = open(file_name, "r")
        list_negative = []
        list_positive = []
        for i in file1.readlines():
            if i[len(i) - 2] == "0":  # if negative review
                i = i[:-3]  # removes the space and 0 at the end
                list_negative.append(i)
            elif i[len(i) - 2] == "1":  # if positive review
                i = i[:-3]  # removes the space and 1 at the end
                list_positive.append(i)
    except IOError:
        print 'File not found!'
        return
    return list_positive, list_negative


def clean_data(my_data):
    for i in range(0, len(my_data)):
        my_data[i] = my_data[i].lower()
        my_data[i] = re.sub('[^A-Za-z0-9\'-]+', ' ', my_data[i])  # removes special characters besides hyphen/apostrophe
        my_data[i] = re.sub('--', ' ', my_data[i])  # removes double hyphen and replacing with space
        my_data[i] = re.sub('[-]', '', my_data[i])  # removes hyphen
        words = my_data[i].split(' ')
        final_string = ''
        for j in words:
            new_str = ''
            matched = False
            match = re.match(r"([a-z']+)([0-9]+)([a-z']+)", j, re.I)
            if not match:  # do nothing for numbers in between letters
                match = re.match(r"([0-9]+)([a-z']+)([0-9]+)", j, re.I)
                if match:
                    new_str += 'num ' + match.groups()[1] + ' num'
                    matched = True
                else:
                    match = re.match(r"([0-9]+)([a-z']+)", j, re.I)  # match/replace numbers with 'num' on only 1 side
                    if match:
                        new_str += 'num ' + match.groups()[1]
                        matched = True
                    else:
                        match = re.match(r"([a-z']+)([0-9]+)", j, re.I)
                        if match:
                            new_str += match.groups()[0] + ' num'
                            matched = True
            else:
                new_str += match.groups()[0] + match.groups()[1] + match.groups()[2]
                matched = True
            if matched:
                final_string += new_str + ' '
            else:
                final_string += j + ' '
        my_data[i] = final_string.strip()
    for i in range(0, len(my_data)):
        my_data[i] = re.sub(r'\b\d+\b', 'num', my_data[i])  # replaces lone numbers with 'num'
    for i in range(0, len(my_data)):  # replaces consecutive numbers with 'num'
        words = my_data[i].split(' ')
        previous_num = False
        new_str = ''
        for word in words:
            if word == '':
                continue
            if word == 'num':
                if not previous_num:
                    new_str += 'num '
                    previous_num = True
            else:
                previous_num = False
                new_str += word + ' '
        my_data[i] = new_str
    return my_data


def calculate_unique_words_freq(train_data, cut_off):
    if cut_off < 0:
        print 'Cutoff must be at least zero!'
        return
    d = []
    for i in range(len(train_data)):
        word = train_data[i].split(' ')
        d += word
    word_counts = Counter(d)  # counter object that returns how many times a word appears
    final_list = word_counts.most_common()  # puts list in order with most occurring words at the beginning
    d = final_list[cut_off:]  # removes most common numbers according to cutOff value
    dictionary = dict(d)
    return dictionary


def calculate_class_probability(pos_train, neg_train):
    return float(len(pos_train))/float(len(pos_train) + len(neg_train)), \
           float(len(neg_train)/float(len(pos_train) + len(neg_train)))


def calculate_scores(class_prob, unique_vocab, test_data):
    list_scores = []
    values = unique_vocab.values()
    denominator = len(unique_vocab) + sum(values)  # denominator of the formula
    for i in test_data:
        word_score_list = []
        arr = i.split(' ')
        for j in arr:
            if j in unique_vocab:
                word_frequency = unique_vocab[j]
            else:
                word_frequency = 0
            word_score_list.append(float(word_frequency + 1) / denominator)  # adds each score to list
        score = class_prob
        for k in word_score_list:  # multiplies the scores together as stated by the given formula
            score *= k
        list_scores.append(score)
    return list_scores


def calculate_accuracy(positive_test_data_positive_scores, positive_test_data_negative_scores,
                       negative_test_data_positive_scores, negative_test_data_negative_scores):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(positive_test_data_negative_scores)):
        if positive_test_data_positive_scores[i] >= positive_test_data_negative_scores[i]:
            tp += 1
        else:
            fp += 1
    for j in range(len(negative_test_data_negative_scores)):
        if negative_test_data_negative_scores[j] > negative_test_data_positive_scores[j]:
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn


def demo(review):
    tup = read_and_divide_by_sentiment("TRAINING.txt")
    positive_train_unique = calculate_unique_words_freq(clean_data(tup[0]), 1)
    negative_train_unique = calculate_unique_words_freq(clean_data(tup[1]), 1)
    class_prob = calculate_class_probability(tup[0], tup[1])
    review_list = clean_data([review])
    positive_scores = calculate_scores(class_prob[0], positive_train_unique, review_list)
    negative_scores = calculate_scores(class_prob[1], negative_train_unique, review_list)
    if positive_scores[0] > negative_scores[0]:
        return 1
    else:
        return -1


def main():
    tup = read_and_divide_by_sentiment("TRAINING.txt")
    tup_test = read_and_divide_by_sentiment("TESTING.txt")
    clean_data(tup[0])
    clean_data(tup[1])
    clean_data(tup_test[0])
    clean_data(tup_test[1])
    review = ' '
    while len(review) > 0:
        review = raw_input('Enter sample review: \n')
        if demo(review) == 1:
            print 'Positive'
        elif demo(review) == -1:
            print 'Negative'
    pos_train_unique = calculate_unique_words_freq(tup[0], 1)
    neg_train_unique = calculate_unique_words_freq(tup[1], 1)
    print neg_train_unique
    class_prob = calculate_class_probability(tup[0], tup[1])
    pos_pos = calculate_scores(class_prob[0], pos_train_unique, tup_test[0])
    pos_neg = calculate_scores(class_prob[1], neg_train_unique, tup_test[0])
    neg_neg = calculate_scores(class_prob[1], neg_train_unique, tup_test[1])
    neg_pos = calculate_scores(class_prob[0], pos_train_unique, tup_test[1])
    print pos_pos
    print pos_neg
    print neg_neg
    print neg_pos

    print calculate_accuracy(pos_pos, pos_neg, neg_pos, neg_neg)


main()
