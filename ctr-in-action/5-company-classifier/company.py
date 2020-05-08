import jieba

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression as LR
import tensorflow as tf

# 根据公司名称和价值点来进行分类
# 名称分词去低频+bow+ sklearn 线性模型，加 n-fold 验证 auc


def filter_char(word):
    filter_words = ['(', ')', '（', '）']
    if word in filter_words:
        return False
    return True

sample_file = './sample.txt'
lines = [line.strip() for line in open(sample_file).readlines()]

company_list = []
quota_list = []
flag_list = []

company_corpus = [] # 实际用来构造向量的 token
word_dict = {}

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
    filtered = []
    for word in words:
        if filter_char(word):
            filtered.append(word)
            word_dict[word] = word_dict.get(word, 0) + 1
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

# 直接用 sklearn 的线性模型
lr = LR(C=1000)
lr.fit(company_vector, flag_list)