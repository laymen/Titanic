#coding: utf-8
import matplotlib.pyplot as plt
import seaborn as sns
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
import numpy as np
import pandas as pd
import re as re

#数据处理，比如参数 SibSp（亲戚和配偶在船数量） 参数Parch（父母孩子的在船数量），合并为有无亲人在船上

train = pd.read_csv('titanic_train.csv', header = 0, dtype={'Age': np.float64})
test  = pd.read_csv('test.csv' , header = 0, dtype={'Age': np.float64})
full_data = [train, test]
#统计家庭成员个数
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

#家庭成员（包括自己）等于一个人的为1，大于1人为0；
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

#用S港口来填充缺失值
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

#用费用的平均值填充缺失值，并将缺失值分成4个范围,pd.acut()函数可以实现该划分
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
#CategoricalFare划分后的结果
#print train

#弥补年龄的缺失值，首先计算现有年龄的平均值，标准方差值范围内随机一个年龄进行填充
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)

train['CategoricalAge'] = pd.cut(train['Age'], 5)
#年龄划分后的结果
#print train

def get_title(name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
	# If the title exists, extract and return it.
	if title_search:
		return title_search.group(1)
	return ""

for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
print full_data

for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
print full_data

for dataset in full_data:
    # 替换性别，女 0，男 1
    dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)

    # 替换称谓
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

    # 替换港口号
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    # 费用化简
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

    # 替换年龄
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4

# 选出我认为重要的特征作为数据集和测试集
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', \
                 'Parch', 'FamilySize']
train = train.drop(drop_elements, axis=1)
train = train.drop(['CategoricalAge', 'CategoricalFare'], axis=1)

test = test.drop(drop_elements, axis=1)



X_train = train.values[:, 1::]
y_train = train.values[:, 0]
X_test = test.values[:, 1::]
y_test = test.values[:, 0]

model_save_file = None
model_save = {}

test_classifiers = ['NB', 'KNN', 'LR', 'RF', 'DT', 'SVM', 'SVMCV', 'GBDT']
classifiers = {'NB': naive_bayes_classifier,
               'KNN': knn_classifier,
               'LR': logistic_regression_classifier,
               'RF': random_forest_classifier,
               'DT': decision_tree_classifier,
               'SVM': svm_classifier,
               'SVMCV': svm_cross_validation,
               'GBDT': gradient_boosting_classifier
               }

print('reading training and testing data...')


for classifier in test_classifiers:
    print('******************* %s ********************' % classifier)
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

'''
classifiers = [
    KNeighborsClassifier(3),
    SVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression()]

log_cols = ["Classifier", "Accuracy"]
log = pd.DataFrame(columns=log_cols)

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

X = train[0::, 1::]
y = train[0::, 0]

acc_dict = {}

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    for clf in classifiers:
        name = clf.__class__.__name__
        clf.fit(X_train, y_train)
        train_predictions = clf.predict(X_test)
        acc = accuracy_score(y_test, train_predictions)
        if name in acc_dict:
            acc_dict[name] += acc
        else:
            acc_dict[name] = acc

for clf in acc_dict:
    acc_dict[clf] = acc_dict[clf] / 10.0
    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
    log = log.append(log_entry)

plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')

sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")

candidate_classifier = SVC()
candidate_classifier.fit(train[0::, 1::], train[0::, 0])
result = candidate_classifier.predict(test)

#测试列表中乘客生还的情况
print result
#各种分类方法的准确率
print acc_dict
'''