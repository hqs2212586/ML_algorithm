# -*- coding:utf-8 -*-
__author__ = 'Qiushi Huang'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linear_regression import LinearRegression

"""单变量线性回归"""

data = pd.read_csv('../data/world-happiness-report-2017.csv')

# 得到训练和测试数据集
train_data = data.sample(frac = 0.8)      # sample：随机选取若干行
test_data = data.drop(train_data.index)   # 将训练数据删除即为测试数据

# 数据和标签定义
input_param_name = "Economy..GDP.per.Capita."
output_param_name = "Happiness.Score"

x_train = train_data[[input_param_name]].values
y_train = train_data[[output_param_name]].values

x_test = test_data[[input_param_name]].values
y_test = test_data[[output_param_name]].values

plt.scatter(x_train, y_train, label="Train data")
plt.scatter(x_test, y_test, label="Test data")
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title('Happy')     # 指定名字
plt.legend()
plt.show()

# 训练线性回归模型
num_iterations = 500
learning_rate = 0.01     # 学习率

linear_regression = LinearRegression(x_train, y_train)   # 线性回归
(theta, cost_history) = linear_regression.train(learning_rate, num_iterations)    # 执行训练

print('开始时的损失', cost_history[0])
print('训练后的损失', cost_history[-1])   # 最后一个

plt.plot(range(num_iterations), cost_history)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('GD')
plt.show()

# 测试
predications_num = 100
x_predications = np.linspace(x_train.min(), x_train.max(), predications_num).reshape(predications_num,1)
y_predications = linear_regression.predict(x_predications)

plt.scatter(x_train, y_train, label="Train data")
plt.scatter(x_test, y_test, label="Test data")
plt.plot(x_predications, y_predications, 'r', label="预测值")
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title("Happy test")
plt.legend()
plt.show()
