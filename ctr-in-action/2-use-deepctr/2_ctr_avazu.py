import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# 通过这样的方式引用上级目录
import sys
sys.path.append("..")
from deepctr.models import *
from deepctr.utils import SingleFeat

if __name__ == "__main__":
    print("读取数据")
    data = pd.read_csv('../data/avazu_train_small')
    split_line = "==================================================="
    # Avazu 的数据都是 categorical 的，所以都是 sparse
    # 获取 column 名称
    sparse_features = data.columns.values.tolist()
    # 去掉 id 和 label
    del sparse_features[0]
    del sparse_features[0]

    # 填充空值 Fill NA/NaN values using the specified method
    # 稀疏的填写 -1，数值的填写 0
    data[sparse_features] = data[sparse_features].fillna('-1', )
    target = ['click']

    print("预览 Sparse Feature")
    print(data[sparse_features].head(10))

    print(split_line)
    print("1. 对 Sparse 特征的 label 做编码")
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    print(split_line)
    print("2. 统计每个 Sparse 特征 Field 中不重复的标签的个数，记录 Dense 特征的 field 名称")
    # 实际就是根据不同的标签个数确定维度，数据越多维度就越大
    sparse_feature_list = [SingleFeat(feat, data[feat].nunique()) for feat in sparse_features ]
    print("Sparse Feature List")
    for sf in sparse_feature_list:
        print(sf)

    print(split_line)
    print("3. 为模型生成输入数据")
    train, test = train_test_split(data, test_size=0.2)
    train_model_input = [train[feat.name].values for feat in sparse_feature_list]
    test_model_input = [test[feat.name].values for feat in sparse_feature_list]

    print(split_line)
    print("4. 定义模型，训练、预测和评估")
    # 这里的 Model 可以换为
    # DeepFM, CCPM, FNN, PNN, MLR, NFM, AFM, xDeepFM
    # AutoInt, FGCNN
    model = FGCNN({
        "sparse": sparse_feature_list},
        task='binary')

    model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'])

    history = model.fit(train_model_input, train[target].values,
                        batch_size=256, epochs=10, verbose=2,
                        validation_split=0.2)
    pred_ans = model.predict(test_model_input, batch_size=256)
    # print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))