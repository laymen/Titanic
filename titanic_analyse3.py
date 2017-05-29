# coding:GBK
__author__ = 'Mouse'
import pandas as pd
import time
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn_test import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

def Time2ISOString(s):
    ''' 把给定的秒转化为定义的格式%Y-%m-%d%H:%M:%S  '''
    str = time.strftime("%p", time.localtime(float(s)))
    return str

def replace_hostname(x):
    if "qq" in x:
        return "qq"
    elif "xunlei" in x:
        return "xunlei"
    elif "taobao" in x:
        return "taobao"
    elif "tanx" in x:
        return "tanx"
    else:
        return "qita"

full_data = pd.read_csv('new_data.csv')
# 选取认为无用的几个特征，username,ip,mac暂时不用
full_data = full_data.drop(['username', 'ip', 'mac', 'hostgrpid'], axis=1)

# 对时间做简单的处理，分为0表示白天，1表示晚上
full_data['time'] = full_data['time'].map(lambda x: 1 if Time2ISOString(x) == 'PM' else 0)
# 对用户浏览的网页进行划分为以下几类，qq baidu xunlei  firefox thinkphp aliyun
# one-hot编码------hostname
full_data['hostname'] = full_data['hostname'].map(lambda x: replace_hostname(x))  # 初步清洗数据
le_hostname = LabelEncoder().fit(full_data['hostname'])
hostname_label = le_hostname.transform(full_data['hostname'])
ohe_hostname = OneHotEncoder(sparse=False).fit(hostname_label.reshape(-1, 1))
hostname_ohe = ohe_hostname.transform(hostname_label.reshape(-1, 1))
full_data['qq'] = hostname_ohe[:, 0]
full_data['xunlei'] = hostname_ohe[:, 1]
full_data['taobao'] = hostname_ohe[:, 2]
full_data['tanx'] = hostname_ohe[:, 3]
full_data['qita'] = hostname_ohe[:, 4]
full_data = full_data.drop('hostname', axis=1)  # 删除原来的hostname数据
# 划分数据集 训练数据 + 测试数据
train = full_data[:int(len(full_data) * 0.6)]
test = full_data[int(len(full_data) * 0.6):]

# 选出我认为重要的特征作为数据集和测试集
# X_train = train.drop('Y_usrgrp_id', axis=1)
# y_train = train['Y_usrgrp_id']
# X_test = test.drop('Y_usrgrp_id', axis=1)
# y_test = test['Y_usrgrp_id']
X_train = train.values[:, 1::]
y_train = train.values[:, 0]
X_test = test.values[:, 1::]
y_test = test.values[:, 0]

model_save_file = None
model_save = {}
test_classifiers = ['NB', 'KNN', 'LR', 'RF', 'DT', 'SVM', 'SVMCV', 'GBDT']
classifiers = {'NB': naive_bayes_classifier,  # 朴素贝叶斯分类
               'KNN': knn_classifier,  # 邻近算法
               'LR': logistic_regression_classifier,  # LR分类器
               'RF': random_forest_classifier,  # 随机森林
               'DT': decision_tree_classifier,  # 决策树
               'SVM': svm_classifier,  # 向量机
               'SVMCV': svm_cross_validation,  # 交叉验证
               'GBDT': gradient_boosting_classifier  # 梯度提升决策树
}
print('reading training and testing data...')
for classifier in test_classifiers:
    print('-------------------- %s -------------------' % classifier)
    start_time = time.time()
    model = classifiers[classifier](X_train, y_train)
    print('training took %fs!' % (time.time() - start_time))
    predict = model.predict(X_test)
    if model_save_file != None:
        model_save[classifier] = model
    # 正确率 = 提取出的正确信息条数 /  提取出的信息条数
    # P = TP/(TP+FP)
    precision = metrics.precision_score(y_test, predict, average='macro')
    # 召回率 = 提取出的正确信息条数 /  样本中的信息条数
    # R = TP/(TP+FN) = 1 - FN/T
    recall = metrics.recall_score(y_test, predict, average='macro')
    print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
    # A = (TP + TN)/(P+N)
    accuracy = metrics.accuracy_score(y_test, predict)
    print('accuracy: %.2f%%' % (100 * accuracy))
