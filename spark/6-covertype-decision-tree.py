from pyspark.context import SparkContext
from pyspark.conf import SparkConf
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.feature import StandardScaler
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.recommendation import Rating, ALS, MatrixFactorizationModel
from pyspark.mllib.evaluation import MulticlassMetrics
from time import time
import pandas as pd

sc = SparkContext(conf=SparkConf().setAppName("MovieLensRec"))

path = "file:/Users/dawang/Documents/GitHub/computational-advertising-note/spark"
print("数据路径", path)
rawData= sc.textFile(path + "/data/covertype/covtype.data")
print("取出前两个")
print(rawData.take(2))
print("清理数据")
lines = rawData.map(lambda x: x.split(","))
print("共计", lines.count(), "项数据")

import numpy as np
def extract_features(record, featureEnd):
    numericalFeatures = [convert_float(field) for field in record[0:featureEnd]]
    return numericalFeatures

def convert_float(x):
    ret =  (0 if x=="?" else float(x))
    return (0 if ret < 0 else ret)

def extract_label(field):
    label = field[-1]
    return float(label)-1 # 从 1~7 调整为 0~6

labelRDD = lines.map(lambda r: extract_label(r))
featureRDD = lines.map(lambda r: extract_features(r, len(r)-1))
labelpointRDD = lines.map(lambda r: LabeledPoint(extract_label(r), extract_features(r, len(r)-1)))
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
    # 这里一定要做一个类型转换
    score = score.map(lambda s: float(s))
    scoreAndLabels = score.zip(validationData.map(lambda p: p.label))
    metrics = MulticlassMetrics(scoreAndLabels)
    accuracy = metrics.accuracy
    return (accuracy)

def trainEvaluateModel(trainData, validationData, impurityParam, maxDepthParam, maxBinsParam):
    startTime = time()
    model = DecisionTree.trainClassifier(trainData, numClasses=7, \
        categoricalFeaturesInfo={}, impurity=impurityParam, \
        maxDepth=maxDepthParam, maxBins=maxBinsParam)
    accuracy = evaluateModel(model, validationData)
    duration = time() - startTime
    print("训练评估：impurity->", impurityParam, ", maxDepth->", maxDepthParam, ", maxBins->", maxBinsParam)
    print("==> 所需时间:", duration, "s , Accuracy=", accuracy)
    return (accuracy, duration, impurityParam, maxDepthParam, maxBinsParam, model)

print("找到最佳参数组合，会计算 2x3x4 = 24 遍")
print("===========================")
impurityList = ["gini", "entropy"]
maxDepthList = [ 20, 25]
maxBinsList = [100, 200]

def evalAllParameter(trainData, validationData, impurityList, maxDepthList, maxBinsList):
    metrics = [trainEvaluateModel(trainData, validationData, impurity, maxDepth, maxBins) \
        for impurity in impurityList 
        for maxDepth in maxDepthList 
        for maxBins in maxBinsList ]
    # 找出 AUC 最大的参数组合
    Smetrics = sorted(metrics, key=lambda k:k[0], reverse=True)
    bestParameter = Smetrics[0]
    print("最佳参数：impurity->", bestParameter[2], ", maxDepth->", bestParameter[3], ", maxBins->", bestParameter[4])
    print("==> 所需时间:", bestParameter[1], "s ,AUC=", bestParameter[0])
    return bestParameter[5]

print("最佳参数组合")
bestModel = evalAllParameter(trainData, validationData, impurityList, maxDepthList, maxBinsList)
print("测试结果")
accuracy = evaluateModel(bestModel, testData)
print("Test Accuracy=", accuracy)