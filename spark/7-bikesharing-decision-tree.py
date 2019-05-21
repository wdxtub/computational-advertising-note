from pyspark.context import SparkContext
from pyspark.conf import SparkConf
from pyspark.mllib.recommendation import Rating, ALS, MatrixFactorizationModel
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.mllib.tree import DecisionTree
from time import time
import pandas as pd

sc = SparkContext(conf=SparkConf().setAppName("MovieLensRec"))

path = "file:/Users/dawang/Documents/GitHub/computational-advertising-note/spark"
print("数据路径", path)
rawDataWithHeader = sc.textFile(path + "/data/bikesharing/hour.csv")
print("取出前两个")
print(rawDataWithHeader.take(2))
print("清理数据")
header = rawDataWithHeader.first()
rawData = rawDataWithHeader.filter(lambda x: x != header)
lines = rawData.map(lambda x: x.split(","))
print(lines.first())
print("共计", lines.count(), "项数据")

import numpy as np
def extract_features(record, featureEnd):
    featureSeason=[convert_float(field) for field in record[2]]
    features = [convert_float(field) for field in record[4: featureEnd-2]]
    # 返回分类特征字段 + 数值特征字段
    return np.concatenate((featureSeason, features))

def convert_float(x):
    return (0 if x=="?" else float(x))

def extract_label(field):
    label = field[-1]
    return float(label)

print("创建 LabeledPoint 数据，用于给决策树进行训练")
from pyspark.mllib.regression import LabeledPoint
labelpointRDD = lines.map(lambda r: LabeledPoint(extract_label(r), extract_features(r,  len(r)-1)))
print("查看下数据")
print(labelpointRDD.take(1))
print("将数据按照 8：1：1 分配，作为训练、验证和测试")
(trainData, validationData, testData) = labelpointRDD.randomSplit([8,1,1])
print("训练", trainData.count(), ",验证", validationData.count(), ",测试", testData.count())
print("数据暂存到内存中")
trainData.persist()
validationData.persist()
testData.persist()

def evaluateModel(model, validationData):
    score = model.predict(validationData.map(lambda p: p.features))
    scoreAndLabels = score.zip(validationData.map(lambda p: p.label))
    metrics = RegressionMetrics(scoreAndLabels)
    return metrics.rootMeanSquaredError

def trainEvaluateModel(trainData, validationData, impurityParam, maxDepthParam, maxBinsParam):
    startTime = time()
    model = DecisionTree.trainRegressor(trainData, categoricalFeaturesInfo={}, \
        impurity=impurityParam, maxDepth=maxDepthParam, maxBins=maxBinsParam)
    RMSE = evaluateModel(model, validationData)
    duration = time() - startTime
    print("训练评估：impurity->", impurityParam, ", maxDepth->", maxDepthParam, ", maxBins->", maxBinsParam)
    print("==> 所需时间:", duration, "s , RMSE=", RMSE)
    return (RMSE, duration, impurityParam, maxDepthParam, maxBinsParam, model)


print("找到最佳参数组合，会计算 2x6x6 = 72 遍")
print("===========================")
impurityList = ["variance"]
maxDepthList = [3, 5, 10, 15, 20, 25]
maxBinsList = [3, 5, 10, 50, 100, 200]

def evalAllParameter(trainData, validationData, impurityList, maxDepthList, maxBinsList):
    metrics = [trainEvaluateModel(trainData, validationData, impurity, maxDepth, maxBins) \
        for impurity in impurityList 
        for maxDepth in maxDepthList 
        for maxBins in maxBinsList ]
    # 找出 AUC 最大的参数组合
    Smetrics = sorted(metrics, key=lambda k:k[0], reverse=False)
    bestParameter = Smetrics[0]
    print("最佳参数：impurity->", bestParameter[2], ", maxDepth->", bestParameter[3], ", maxBins->", bestParameter[4])
    print("==> 所需时间:", bestParameter[1], "s ,RMSE=", bestParameter[0])
    return bestParameter[5]

print("最佳参数组合")
bestModel = evalAllParameter(trainData, validationData, impurityList, maxDepthList, maxBinsList)
print("测试结果")
RMSE = evaluateModel(bestModel, testData)
print("Test RMSE=", RMSE)