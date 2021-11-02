# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 11:07:00 2021
"""
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import export_graphviz

font = font_manager.FontProperties(fname = r'C:\Windows\Fonts\times.ttf', size = 10)

titanic_train = pd.read_csv(r'.\kaggle-titanic\train.csv')
titanic_test = pd.read_csv(r'.\kaggle-titanic\test.csv')
titanic_test_result = pd.read_csv(r'.\kaggle-titanic\gender_submission.csv')
titanic_test = pd.merge(titanic_test, titanic_test_result, on = 'PassengerId')

#训练集的特征和目标
features = titanic_train[['Pclass', 'Age', 'Sex']] #特征值
targets = titanic_train[['Survived']] #目标值

#测试集的特征和目标
features_test = titanic_test[['Pclass', 'Age', 'Sex']] #特征值
targets_test = titanic_test[['Survived']] #目标值

#数据清洗：用平均值来代替，并用年龄的平均值来替换。inplace如果不是True则只会返回结果并不会覆盖原值
features['Age'].fillna(features['Age'].mean(), inplace = True)
features_test['Age'].fillna(features_test['Age'].mean(), inplace = True)

#X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size = 0.25) #切分训练集与测试集

#特征提取部分
vect = DictVectorizer()
features = vect.fit_transform(features.to_dict(orient = 'records'))
features_test = vect.fit_transform(features_test.to_dict(orient = 'records'))

#决策树算法
classifier = DecisionTreeClassifier()
classifier.fit(features, targets)
print(classifier.score(features_test, targets_test))

#预测部分:我们在预测的同时验证classifier.score的值事实上就是准确率
arr = features_test.toarray()
accu = 0
for idx in range(arr.shape[0]):
    sample = features_test[idx]
    if classifier.predict(sample)[0] == targets_test.iloc[idx, 0]:
        accu += 1
print('Accuracy:' + str(accu/(arr.shape[0])))

#决策树可视化
export_graphviz(classifier, 'tree_class.dot', 
                feature_names = vect.get_feature_names(),
                class_names = ['died', 'survived'])
