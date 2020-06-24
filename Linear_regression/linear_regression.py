# -*- coding:utf-8 -*-
__author__ = 'Qiushi Huang'

import numpy as np
from utils.features.prepare_for_training import prepare_for_training

"""线性回归"""
class LinearRegression:
    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
        """
        1.对数据进行预处理操作
        2.先得到所有的特征个数
        3.初始化参数矩阵
        """
        (
            data_processed,
            features_mean,
            features_deviation
        ) = prepare_for_training(data, polynomial_degree, sinusoid_degree, normalize_data)

        self.data = data_processed
        self.labels = labels
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data

        # 获取多少个列作为特征量
        num_features = self.data.shape[1]    # 1是列个数，0是样本个数
        self.theta = np.zeros((num_features, 1))   # 构建θ矩阵

    def train(self, alpha, num_iterations=500):
        """
        训练模块，执行梯度下降
        :param alpha: α为学习率（步长）
        :param num_iterations: 迭代次数
        :return:
        """
        cost_history = self.gradient_descent(alpha, num_iterations)
        return self.theta, cost_history

    def gradient_descent(self, alpha, num_iterations):
        """
        梯度下降，实际迭代模块
        :param alpha: 学习率
        :param num_iterations: 迭代次数
        :return:
        """
        cost_history = []     # 保存损失值
        for _ in range(num_iterations):   # 每次迭代参数更新
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data, self.labels))
        return cost_history

    def gradient_step(self, alpha):
        """
        梯度下降参数更新计算方法（核心代码,矩阵运算）
        :param alpha: 学习率
        :return:
        """
        num_examples = self.data.shape[0]           # 样本数
        prediction = LinearRegression.hypothesis(self.data, self.theta)           # 预测值
        # 参差=预测值-真实值
        delta = prediction - self.labels
        theta = self.theta
        # theta值更新，.T是执行转置
        theta = theta - alpha * (1/num_examples)*(np.dot(delta.T, self.data)).T
        self.theta = theta

    def cost_function(self, data, labels):
        """
        损失计算
        :param data: 数据集
        :param labels: 真实值
        :return:
        """
        num_examples = data.shape[0]       # 样本个数
        # 参差=预测值-真实值
        delta = LinearRegression.hypothesis(self.data, self.theta) - labels
        # Calculate current predictions cost.
        cost = (1/2)*np.dot(delta.T, delta)
        # print(cost.shape)
        # print(cost)
        return cost[0][0]

    @staticmethod
    def hypothesis(data, theta):
        """
        预测函数
        :param data:
        :param theta:
        :return:
        """
        # 如果处理的是一维数组，则得到的是两数组的內积；如果处理的是二维数组（矩阵），则得到的是矩阵积
        predictions = np.dot(data, theta)
        return predictions

    def get_cost(self, data, labels):
        """
        计算当前损失
        :param data:
        :param labels:
        :return:
        """
        # 经过处理了的数据
        data_processed = prepare_for_training(data,
                                              self.polynomial_degree,
                                              self.sinusoid_degree,
                                              self.normalize_data)[0]
        return self.cost_function(data_processed, labels)    # 返回损失值

    def predict(self, data):
        """
        用训练的数据模型，预测得到回归值结果
        :param data:
        :return:
        """
        # 经过处理了的数据
        data_processed = prepare_for_training(data,
                                              self.polynomial_degree,
                                              self.sinusoid_degree,
                                              self.normalize_data)[0]
        predictions = LinearRegression.hypothesis(data_processed, self.theta)
        return predictions      # 返回预测值







