import jieba

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
from sklearn.linear_model import LogisticRegression, LinearRegression, LogisticRegressionCV
import tensorflow as tf
from sklearn.svm import SVC

# 根据公司名称和价值点来进行分类
# 名称分词去低频+bow+ sklearn 线性模型，加 n-fold 验证 auc

word_dict = {}

def filter_char(word):
    filter_words = ['(', ')', '（', '）']
    if word in filter_words:
        return False
    if word_dict[word] < 2:
        return False
    return True

sample_file = './sample.txt'
lines = [line.strip() for line in open(sample_file).readlines()]

company_list = []
quota_list = []
flag_list = []

company_corpus = [] # 实际用来构造向量的 token


for line in lines:
    arr = line.split(' ')
    company = arr[0]
    quota = 0
    flag = 0
    if len(arr) > 1: # 没有第二个值
        quota = float(arr[1])
        flag = 1
    company_list.append(company)
    quota_list.append(quota)
    flag_list.append(flag)

# build dict
for company in company_list:
    words = jieba.lcut(company)
    for word in words:
        word_dict[word] = word_dict.get(word, 0) + 1

# filter word with freq 1
for company in company_list:
    words = jieba.lcut(company)
    filtered = []
    for word in words:
        if filter_char(word):
            filtered.append(word)
    # for CountVectorizer
    tokenizer = " ".join(filtered)
    company_corpus.append(tokenizer)

print('word dict -----')
items = list(word_dict.items())
items.sort(key=lambda x:x[1], reverse=True)
for word, count in items:
    print(word, count)
print('word dict length', len(word_dict.keys()))

# bag of words
vectorizer = CountVectorizer(analyzer='word', token_pattern=u"(?u)\\b\\w+\\b")
doc = vectorizer.fit_transform(company_corpus)
print('bag of words')
company_vector = doc.toarray() # 这个就是 X
print(company_vector[0])
print('dictionary word count', len(vectorizer.get_feature_names()))
print(vectorizer.get_feature_names())

X = np.array(company_vector)
Y = np.array(flag_list)

# 直接用 sklearn 的线性模型
# solver
# liblinear 使用了坐标轴下降法来迭代优化损失函数
# lbfgs 拟牛顿法的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数
# newton-cg 也是牛顿法法的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数
# sag 即随机平均梯度下降，是梯度下降法的变种，是一种线性收敛算法，和普通梯度下降法的区别是每次迭代仅仅用一部分的样本来计算梯度，适合于样本数据多的时候
lr = LogisticRegression(C=1000, solver='liblinear')

# 测试 Kfold
kf = KFold(n_splits=3, shuffle=True)
for train_index, test_index in kf.split(X):
    print('train_index: %s, test_index: %s' % (train_index, test_index))
    lr.fit(X[train_index], Y[train_index])
    score = lr.score(X[train_index], Y[train_index])
    print('train score', score)
    y_pred = lr.predict(X[test_index])
    y_pred_prob = lr.predict_proba(X[test_index])
    print('y_pred', y_pred)
    print('y_pred_prob', y_pred_prob)
    print('confusion_matrix')
    print(metrics.confusion_matrix(Y[test_index], y_pred))
    print('auc', metrics.roc_auc_score(Y[test_index], y_pred))


Y = np.array(quota_list)

# lr = LinearRegression()
# # 测试 Kfold
# kf = KFold(n_splits=3, shuffle=True)
# for train_index, test_index in kf.split(X):
#     print('train_index: %s, test_index: %s' % (train_index, test_index))
#     lr.fit(X[train_index], Y[train_index])
#     score = lr.score(X[train_index], Y[train_index])
#     print('train score', score)
#     y_pred = lr.predict(X[test_index])
#     print('y_pred', y_pred)
#     print('mean absolute_error', metrics.mean_absolute_error(Y[test_index], y_pred))
#     for i in range(len(y_pred)):
#         print('y_true', Y[test_index[i]], 'y_pred', y_pred[i])


# svm = SVC()
# # 测试 Kfold
# kf = KFold(n_splits=3, shuffle=True)
# for train_index, test_index in kf.split(X):
#     print('train_index: %s, test_index: %s' % (train_index, test_index))
#     svm.fit(X[train_index], Y[train_index])
#     score = svm.score(X[train_index], Y[train_index])
#     print('train score', score)
#     test_score = svm.score(X[test_index], Y[test_index])
#     print('test score', test_score)
#     y_pred = svm.predict(X[test_index])
#     print(y_pred)