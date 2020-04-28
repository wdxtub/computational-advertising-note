import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler

LEARNING_RATE = 0.01
BATCH_SIZE = 32
EPOCH = 30


if __name__ == "__main__":
    print("读取数据")
    data = pd.read_csv('../data/criteo_train_small.txt')
    split_line = "==================================================="
    print('查看数据维度')
    print(data.info())
    print('查看前 5 行数据')
    print(data.head(5))
    print('稀疏标签空值补 -1，连续标签空值补 0')
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    sparse_dict = {}
    total_dimension = 0
    dense_features = ['I' + str(i) for i in range(1, 14)]
    # 目标值
    target = ['label']
    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    print("预览 Sparse Feature")
    print(data[sparse_features].head(5))
    print("预览 Dense Feature")
    print(data[dense_features].head(5))

    print(split_line)
    print("1. 对 Sparse 特征的 label 做编码，把 Dense 变换到 0 1 之间")
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    print("处理后的 Sparse Feature")
    print(data[sparse_features].head(5))

    print("离散化 Dense Feature Onehot 编码")
    for key in dense_features:
        data[key] = pd.qcut(data[key], 10, duplicates='drop', labels=False)
        data[key] = data[key].astype('float64')
    print(data[dense_features].head(5))

    # 可以比较一下非 onehot 与 onehot 效果差别，明显 one-hot 后有进步！

    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    hot = ohe.fit_transform(data[dense_features])
    dense_data = pd.DataFrame(hot)
    # 拼接
    dense_data['label'] = data[target]

    print(dense_data.head(5))
    dimension = len(hot[0])

    hot_features = [i for i in range(0, dimension)]


    # 权值
    W = tf.get_variable('weights',
                        dtype=tf.float64,
                        shape=[dimension, 1],
                        regularizer=tf.contrib.layers.l2_regularizer(0.02))
    #W = tf.Variable(tf.zeros([dimension, 1], dtype=tf.float64), name='weights')
    b = tf.Variable(tf.zeros([1], dtype=tf.float64), name='bias')


    def lr_model(inputs):
        return tf.sigmoid(tf.matmul(inputs, W) + b)


    print(split_line)
    print("2. 统计每个 Sparse 特征 Field 中不重复的标签的个数，保存到字典中")
    for feat in sparse_features:
        count = data[feat].nunique()
        sparse_dict[feat] = count
        total_dimension += count
    print(sparse_dict)
    # 41 万，太宽太稀疏，所以需要做 embedding
    print(f'One-hot 编码离散特征维度: {total_dimension}')

    print(split_line)
    print("3. 为模型生成输入数据，先只用稠密矩阵作为输入")
    print("切分数据为训练集与测试集")
    train, test = train_test_split(dense_data, test_size=0.2)
    print("转为 TF Dataset")
    train_dataset = tf.data.Dataset.from_tensor_slices((train[hot_features].values, train[target].values))
    train_dataset = train_dataset.shuffle(3000).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices((test[hot_features].values, test[target].values))
    # train_dataset = tf.data.Dataset.from_tensor_slices((train[dense_features].values, train[target].values))
    # train_dataset = train_dataset.shuffle(3000).batch(BATCH_SIZE)
    # test_dataset = tf.data.Dataset.from_tensor_slices((test[dense_features].values, test[target].values))

    train_iter = train_dataset.make_initializable_iterator()
    test_iter = test_dataset.make_initializable_iterator()
    X, Y = train_iter.get_next()

    yp = lr_model(X)
    Y = tf.cast(Y, dtype=tf.float64)

    # reduce_sum + onehot = 0.7215
    # reduce_mean + onehot = 0.7227
    # reduce_mean + onehot + l1(0.01) = 0.725
    # reduce_mean + onehot + l2(0.01) = 0.7252 差别不大
    # 和 w 初始值非常大关系，很看运气
    tf.add_to_collection('cost', tf.contrib.layers.l2_regularizer(0.02)(W))
    loss = -tf.reduce_mean(Y * tf.log(yp) + (1-Y) * tf.log(1-yp))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

    # 获取测试的预测值，并进行比较
    # tp = lr_model(test[dense_features].values)
    tp = lr_model(test[hot_features].values)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(EPOCH):
            sess.run(train_iter.initializer)
            total_loss = 0
            try:
                while True:
                    _, _loss = sess.run([optimizer, loss])
                    total_loss += _loss
            except tf.errors.OutOfRangeError:
                pass
            _tp = sess.run([tp])
            print('Loss Epoch {0}: {1}, test AUC {2}'.format(i,
                                                             total_loss,
                                                             round(roc_auc_score(test[target].values, _tp[0]), 4)))



