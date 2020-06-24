# -*- coding:utf-8 -*-
__author__ = 'Qiushi Huang'

"""数据标准化"""

import numpy as np

def normalize(features):
    """Normalize features.
    Normalizes input features X. Returns a normalized version of X where the mean value of
    each feature is 0 and deviation is close to 1.
    :param features: set of features.
    :return: normalized set of features.
    """

    # astype：转换数组的数据类型
    features_normalized = np.copy(features).astype(float)

    # 计算均值
    features_mean = np.mean(features, 0)

    # 计算标准差
    features_deviation = np.std(features, 0)

    # 标准化操作
    if features.shape[0] > 1:
        features_normalized -= features_mean

    # 防止除以0
    features_deviation[features_deviation == 0] = 1
    features_normalized /= features_deviation

    return features_normalized, features_mean, features_deviation