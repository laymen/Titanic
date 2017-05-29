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
    ''' �Ѹ�������ת��Ϊ����ĸ�ʽ%Y-%m-%d%H:%M:%S  '''
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
# ѡȡ��Ϊ���õļ���������username,ip,mac��ʱ����
full_data = full_data.drop(['username', 'ip', 'mac', 'hostgrpid'], axis=1)

# ��ʱ�����򵥵Ĵ�����Ϊ0��ʾ���죬1��ʾ����
full_data['time'] = full_data['time'].map(lambda x: 1 if Time2ISOString(x) == 'PM' else 0)
# ���û��������ҳ���л���Ϊ���¼��࣬qq baidu xunlei  firefox thinkphp aliyun
# one-hot����------hostname
full_data['hostname'] = full_data['hostname'].map(lambda x: replace_hostname(x))  # ������ϴ����
le_hostname = LabelEncoder().fit(full_data['hostname'])
hostname_label = le_hostname.transform(full_data['hostname'])
ohe_hostname = OneHotEncoder(sparse=False).fit(hostname_label.reshape(-1, 1))
hostname_ohe = ohe_hostname.transform(hostname_label.reshape(-1, 1))
full_data['qq'] = hostname_ohe[:, 0]
full_data['xunlei'] = hostname_ohe[:, 1]
full_data['taobao'] = hostname_ohe[:, 2]
full_data['tanx'] = hostname_ohe[:, 3]
full_data['qita'] = hostname_ohe[:, 4]
full_data = full_data.drop('hostname', axis=1)  # ɾ��ԭ����hostname����
# �������ݼ� ѵ������ + ��������
train = full_data[:int(len(full_data) * 0.6)]
test = full_data[int(len(full_data) * 0.6):]

# ѡ������Ϊ��Ҫ��������Ϊ���ݼ��Ͳ��Լ�
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
classifiers = {'NB': naive_bayes_classifier,  # ���ر�Ҷ˹����
               'KNN': knn_classifier,  # �ڽ��㷨
               'LR': logistic_regression_classifier,  # LR������
               'RF': random_forest_classifier,  # ���ɭ��
               'DT': decision_tree_classifier,  # ������
               'SVM': svm_classifier,  # ������
               'SVMCV': svm_cross_validation,  # ������֤
               'GBDT': gradient_boosting_classifier  # �ݶ�����������
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
    # ��ȷ�� = ��ȡ������ȷ��Ϣ���� /  ��ȡ������Ϣ����
    # P = TP/(TP+FP)
    precision = metrics.precision_score(y_test, predict, average='macro')
    # �ٻ��� = ��ȡ������ȷ��Ϣ���� /  �����е���Ϣ����
    # R = TP/(TP+FN) = 1 - FN/T
    recall = metrics.recall_score(y_test, predict, average='macro')
    print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
    # A = (TP + TN)/(P+N)
    accuracy = metrics.accuracy_score(y_test, predict)
    print('accuracy: %.2f%%' % (100 * accuracy))
