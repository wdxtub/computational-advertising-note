import jieba

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
import tensorflow as tf

def filter_char(word):
    filter_words = ['(', ')', '（', '）', '有限公司', '有限责任', '公司']
    if word in filter_words:
        return False
    return True

sample_file = './sample.txt'
lines = [line.strip() for line in open(sample_file).readlines()]
# build dict
brand2company = {}
company2brand = {}
word_dict = {}
corpus = []
brand_list = []
company_list = []
y_list = []

for line in lines:
    arr = line.split(' ')
    brand = arr[0]
    company = arr[1]
    y_label = float(arr[2])
    y_list.append(y_label)

    brand_list.append(brand)
    company_list.append(company)

    if brand2company.get(brand) is not None:
        brand2company[brand].append(company)
    else:
        brand2company[brand] = [company]
    
    if company2brand.get(company) is not None:
        if brand not in company2brand[company]:
            company2brand[company].append(brand)
    else:
        company2brand[company] = [brand]

print('stats --------')
print(f'unique company #{len(company2brand.keys())}, unique brand #{len(brand2company.keys())}')
for key in brand2company.keys():
    print(f'brand {key} has {len(brand2company[key])} company: {brand2company[key]}')
print('- - - - ')
for key in company2brand.keys():
    print(f'company {key} delegate {len(company2brand[key])} brand: {company2brand[key]}')
    
for company in company_list:
    words = jieba.lcut(company)
    filtered = []
    for word in words:
        if filter_char(word):
            filtered.append(word)
            word_dict[word] = word_dict.get(word, 0) + 1
    # for CountVectorizer
    tokenizer = " ".join(filtered)
    corpus.append(tokenizer)

print('word dict -----')
items = list(word_dict.items())
items.sort(key=lambda x:x[1], reverse=True)
for word, count in items:
    print(word, count)
print('word dict length', len(word_dict.keys()))

# keep single word
vectorizer = CountVectorizer(analyzer='word', token_pattern=u"(?u)\\b\\w+\\b")
doc = vectorizer.fit_transform(corpus)
print('bag of words')
# 这里公司名字 bow 向量已 ok
name_vector = doc.toarray()
print(name_vector[0])
print(name_vector.shape)

print('dictionary word count', len(vectorizer.get_feature_names()))
print(vectorizer.get_feature_names())

# 准备 品牌 onehot，不需要先 label（但是在 wtss 中需要，因为 sklearn 版本不同）
brands =  np.array(brand_list) 
ohe = OneHotEncoder(sparse=False).fit(brands.reshape(-1,1))
oh_brand_label = ohe.transform(brands.reshape(-1,1))
print('brands', brands)
# wtss 上没有这个 api
print('label dict', ohe.get_feature_names())
print('onehot brad label', oh_brand_label)
print(oh_brand_label.shape)
# 拼接
dataX = np.concatenate([doc.toarray(), oh_brand_label], axis=1)
print(dataX[0])

# 区分训练和测试集
train_x, test_x, train_y, test_y = train_test_split(dataX, np.array(y_list), test_size=0.2)
print('train_x shape', train_x.shape)
print('test_x shape', test_x.shape)

# 构造图并训练
n, p = train_x.shape
k = 10 # num of latent factor
X = tf.placeholder('float', shape=[None, p])
Y = tf.placeholder('float', shape=[None, 1])

# bias and weights
w0 = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.zeros([p]))

# matrix factorization factors, randomly initialized
V = tf.Variable(tf.random_normal([k, p], stddev=0.01))

# estimation of y
Y_hat = tf.Variable(tf.zeros([n, 1]))

# 定义损失函数和优化器
linear_terms = tf.add(w0, tf.reduce_sum(tf.multiply(W, X), 1, keepdims=True))
pair_interactions = tf.multiply(0.5,
                                tf.reduce_sum(
                                    tf.subtract(
                                        tf.pow(tf.matmul(X, tf.transpose(V)), 2),
                                        tf.matmul(tf.pow(X, 2), tf.pow(tf.transpose(V), 2))
                                    ), 1, keepdims=True
                                ))
Y_hat = tf.add(linear_terms, pair_interactions)
lambda_w = tf.constant(0.001, name='lambda_w')
lambda_v = tf.constant(0.001, name='lambda_v')
l2_norm = tf.reduce_sum(tf.multiply(lambda_w, tf.pow(W,2)))+tf.reduce_sum(tf.multiply(lambda_v, tf.pow(V,2)))
error = tf.reduce_mean(tf.square(tf.subtract(Y, Y_hat)))
loss =  tf.add(error, l2_norm)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

# 训练和测试
epochs = 50
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for e in range(epochs):
        _, loss = sess.run([optimizer, error], feed_dict={
            X: train_x.reshape(-1, p),
            Y: train_y.reshape(-1, 1)
        })
        print(f'Epoch {e} Loss {loss}')

    # 尝试预测一下
    predict = sess.run(Y_hat, feed_dict={X: test_x})
    for i in range(len(test_y)):
        print(f'predict: {predict[i]}, true: {test_y[i]}')
    auc = metrics.roc_auc_score(test_y, predict)
    print('auc', auc)
