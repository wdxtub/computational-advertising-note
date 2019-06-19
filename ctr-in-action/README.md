# CTR 学习

基于 DeepCTR 这个项目来操作

## 版本

只列举比较关键的包

+ python 3.6.8
+ tensorflow 1.12.0
+ numpy 1.16.4
+ pandas 0.24.2
+ scikit-learn 0.21.2
+ scipy 1.3.0
+ request 2.22.0
+ keras 2.2.4

## 数据集

+ 对于 Categorical 的特征，都属于 Sparse Feature
+ 对于数值型特征，都属于 Dense Feature

### Criteo 数据集

```bash
# 训练数据行数 45840617（千万级）
wc -l criteo_train.txt
# 测试数据行数 6042135（百万级）
wc -l criteo_test.txt
# 用 1_explore_split_critoe.py 来分离数据
```

在 200000 条数据上的表现，模型参数为默认，训练参数：batch_size=256, epochs=10, verbose=2, validation_split=0.2

+ DeepFM: AUC 0.702，一个 epoch 20s
+ CCPM: AUC 0.6818，一个 epoch 30s
+ FNN: AUC 0.7015，一个 epoch 30s
+ PNN: AUC 0.703，一个 epoch 20s
+ MLR: AUC 0.7584，一个 epoch 20s
+ NFM: AUC 0.6789，一个 epoch 20s
+ AFM: AUC 0.7597，一个 epoch 40s
+ DCN: AUC 0.6839，比较慢，一个 epoch 200s
+ xDeepFM: AUC 0.6997，比较慢，一个 epoch 100s
+ AutoInt AUC 0.7027，一个 epoch 35s
+ NFFM: 非常慢，不纳入比较
+ FGCNN: AUC 0.7115，比较慢，一个 epoch 203s

### Avazu 数据集

```bash
# 训练数据行数 40428968（千万级）
wc -l avazu_train
# 测试数据行数 4577465（百万级）
wc -l avazu_test
# 用 1_explore_split_critoe.py 来分离数据
```

在 400000 条数据上的表现，模型参数为默认，训练参数：batch_size=256, epochs=10, verbose=2, validation_split=0.2

+ DeepFM: AUC 0.7472，一个 epoch 25s
+ CCPM: AUC 0.7452，一个 epoch 40s
+ FNN: AUC 0.7437，一个 epoch 25s
+ PNN: AUC 0.7313，一个 epoch 30s
+ NFM: AUC 0.7337，一个 epoch 25s
+ AFM: AUC 0.7619，一个 epoch 30s
+ DCN: AUC 0.7344，比较慢，一个 epoch 275s
+ xDeepFM: AUC 0.7374，比较慢，一个 epoch 85s
+ AutoInt AUC 0.7394，一个 epoch 40s
+ FGCNN: AUC 0.7318，比较慢，一个 epoch 165s
+ NFFM: 非常慢，不纳入比较



## 参考

+ [repo:shenweichen/DeepCTR](https://github.com/shenweichen/DeepCTR)
+ https://pnyuan.github.io/blog/ml_practice/Kaggle%E6%BB%91%E6%B0%B4%20-%20CTR%E9%A2%84%E4%BC%B0%EF%BC%88LR%EF%BC%89/