#!/usr/bin/env python
# encoding: utf-8
# the method is derived from https://blog.csdn.net/qq_39783265/article/details/99712652

from sklearn import svm
import numpy as np
from memory_profiler import profile
import time

time_begin = time.time()


# @profile(precision=4, stream=open('memory_profiler_SVM.log','w+'))
def my_func():
# ------------------------------- Main Function ------------------------------------------------------------------------
    file_path = './MachineLearningCSV/MachineLearningCVE/data_train.txt'
    data = np.loadtxt(file_path, dtype=float, delimiter=',')
    # print(data.shape)
    x, y = np.split(data, (78, ), axis=1)
    x_train = x
    y_train = y.ravel()

    classifier = svm.SVC(kernel='rbf', gamma=0.04, decision_function_shape='ovo', probability=True)
    classifier.fit(x_train, y_train)

    test_file_path = './MachineLearningCSV/MachineLearningCVE/data_test.txt'
    data_test = np.loadtxt(test_file_path, dtype=float, delimiter=',')
    x_test, y_test = np.split(data_test, (78, ), axis=1)
    y_test = y_test.ravel()

    length, width = x_test.shape


    # ------------------------- accuracy calculation -------------------------------------------------------------------
    n = 0
    for i in range(length):
        # if i >= 5:
        #     break
        test = x_test[i, :]
        test = test.reshape(1, 78)
        result = classifier.predict(test)
        label = y_test[i]
        if result[0] == label:
            n += 1

    print('The accuracy is: %0.5f' % (n/length))


    # ------------------------- precise probability --------------------------------------------------------------------
    order = 2
    test = x_test[order-1, :]
    test = test.reshape(1, 78)
    label = y_test[order-1]

    result = classifier.predict(test)
    prob = classifier.predict_proba(test)
    print(result[0], label)
    print('The malignant intrusion probability is: %0.6f' % prob[0, 1])





if __name__ == '__main__':
    my_func()

    time_end = time.time()
    time = time_end - time_begin
    print('Time is about: %0.8f seconds' % time)
    # print(hpy().heap())   # 内存记录
















# -------------------------- Iris_data sample --------------------------------------------------------------------------
# def iris_type(s):
#     class_label={b'setosa': 0, b'versicolor': 1, b'virginica': 2} # b, 表示python解码的标志位
#     return class_label[s]

# def train(model, x_train, y_train):
#     model.fit(x_train, y_train.ravel())


# if __name__ == '__main__':
#     file_path = './iris/iris.data'
#     data = np.loadtxt(file_path, dtype=float, delimiter=',', converters={4:iris_type})
#     # print(data)
#     x,y = np.split(data, (4,), axis=1)
#     # print(x, y)
#     x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, random_state=1, test_size=0.3)
#     y_train = y_train.ravel()
#     # print(x_train)
#     print(y_train.ravel())     # 降维函数
#
#     classifier = svm.SVC(kernel='rbf', gamma= 0.1, decision_function_shape='ovo', C=0.8, probability=True)
#     # kernel = 'rbf'（default）时，为高斯核函数，gamma值越小，分类界面越连续；gamma值越大，分类界面越“散”，分类效果越好，但有可能会过拟合。
#     # decision_function_shape='ovo'时，为one v one分类问题，即将类别两两之间进行划分，用二分类的方法模拟多分类的结果。
#     # decision_function_shape='ovr'时，为one v rest分类问题，即一个类别与其他类别进行划分。
#
#     classifier.fit(x_train, y_train) # x 为 二维数组 y 为一维数组
#
#     test = x_test[2, :]
#     test = test.reshape(1, 4)
#     print(test)
#
#     result = classifier.predict(test)
#     prob = classifier.predict_proba(test)
#     print(result)
#     print(prob.ravel())









