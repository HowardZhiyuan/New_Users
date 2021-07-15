#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import scipy.special
import time
from memory_profiler import profile

# the set of train contains 60 times of 200 lists

class NeuralNetwork:

    # 对象实例化
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # print('initializing')
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate

        # 链接权重矩阵 wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the layer
        # w11 w21
        # w12 w22  etc.

        # --------- 简单地权重设置 -------------------------------------
        # self.wih = np.random.rand(self.hnodes,self.inodes)-0.5
        # self.who = np.random.rand(self.onodes,self.hnodes)-0.5

        # --------- 复杂的权重设置 -------------------------------------
        # 采用正态分布概率采样权重，平均值为0，标准方差为节点传入链接数目的开方，即1/hidden_nodes^(0.5)
        self.wih = np.random.normal(0, pow(self.hnodes, -0.5), (self.hnodes,self.inodes))
        self.who = np.random.normal(0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # 激活函数 sigmoid(x)
        self.activation_function = lambda x: scipy.special.expit(x)

    # 训练模型
    def train(self, input_list, target_list):
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T

        # 计算每个训练样本的输出值
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # 修改权重，训练模型
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)
        delta_wjk = np.dot(self.lr * output_errors * final_outputs * (1 - final_outputs), np.transpose(hidden_outputs))
        self.who = self.who + delta_wjk
        delta_wij = np.dot(self.lr * hidden_errors * hidden_outputs * (1 - hidden_outputs), np.transpose(inputs))
        self.wih = self.wih + delta_wij

        return self.who, self.wih

    # 在完成训练之后，权重参数已经给定，再调用该模型来预测测试集的输入
    def query(self, input_list):
        inputs = np.array(input_list, ndmin=2).T                     # 生成一个2维的列向量
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
time_begin = time.time()

# @profile(precision=4, stream=open('memory_profiler_BPANN.log', 'w+'))
def my_function():

    input_nodes = 40
    hidden_nodes = 35
    output_nodes = 2
    learning_rate = 0.08

    NN = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # load training data
    training_data = open('./MachineLearningCSV/MachineLearningCVE/data_train_new.txt', 'r')
    data_set = training_data.readlines()
    training_data.close()


    # targets = open('label_feature.txt', 'r')
    # data_label = targets.readlines()
    # print(data_label)
    # print(data_set)
    # num = 0
    # label = np.asfarray(data_label)
    # label = label - 1
    # print(label)

    # ----------------------------------- train model -------------------------------------
    epochs = 3
    for e in range(epochs):
        for data in data_set:
            # num += 1
            # if num > 5:
            #     break
            data = data.strip() # 删除 \n
            value = data.split(',')
            inputs = np.asfarray(value[:-1]) + 0.001
            targets = np.zeros(output_nodes) + 0.001
            order = value[-1:]
            targets[int(order[0])-1] = 0.99
            NN.train(inputs, targets)


    #  ------------------------------------- test model -----------------------------------------
    test_data = open('./MachineLearningCSV/MachineLearningCVE/data_test_new.txt', 'r')
    test_set = test_data.readlines()
    test_data.close()

    # -------------------------------------- accuracy calculations ------------------------------
    n = 0
    scoreboard = 0

    for inputdata in test_set:
        n += 1
        inputdata = inputdata.strip()
        value = inputdata.split(',')
        test_inputs = np.asfarray(value[:-1]) + 0.001

        result = NN.query(test_inputs)

        index = np.argmax(result)
        index = index.tolist()
        order = value[-1:]

        label = int(order[0])-1
        # print(index, label)
        # print(type(index), type(label))
        if index == label:
            scoreboard += 1
        else:
            pass

    print('The accuracy is: %0.4f' % (scoreboard/n*100))

    # -------------------------------------- precise probability ------------------------------

    test_data = np.loadtxt('./MachineLearningCSV/MachineLearningCVE/data_test_new.txt', dtype=float, delimiter=',')
    order = 4
    x, y = np.split(test_data, (40, ), axis=1)
    x_test = x[order-1, :]
    y = y.ravel()
    label = y[order-1]

    value = x_test + 0.001
    result = NN.query(value)
    index = np.argmax(result)
    index = index.tolist() + 1

    print(index, label)
    print('the malignant intrusion probability is: %0.6f' % result[1, 0])

if __name__ == '__main__':
    my_function()
    time_end = time.time()
    time = time_end - time_begin
    print('Time is about: %0.8f seconds' % time)
