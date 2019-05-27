from pyspark.context import SparkContext
from pyspark.conf import SparkConf
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.feature import StandardScaler
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.recommendation import Rating, ALS, MatrixFactorizationModel
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from time import time
import pandas as pd

sc = SparkContext(conf=SparkConf().setAppName("MovieLensRec"))

path = "file:/Users/dawang/Documents/GitHub/computational-advertising-note/spark"
print("数据路径", path)
rawDataWithHeader = sc.textFile(path + "/data/stumbleupon/train.tsv")
print("取出前两个")
print(rawDataWithHeader.take(2))
print("清理数据")
header = rawDataWithHeader.first()
rawData = rawDataWithHeader.filter(lambda x: x != header)
rData = rawData.map(lambda x: x.replace("\"", ""))
lines = rData.map(lambda x: x.split("\t"))
print("共计", lines.count(), "项数据")

import numpy as np
def extract_features(field, categoriesMap, featureEnd):
    # 提取分类特征字段
    categoryIdx = categoriesMap[field[3]]
    categoryFeatures = np.zeros(len(categoriesMap))
    categoryFeatures[categoryIdx] = 1
    # 提取数值字段
    numericalFeatures = [convert_float(field) for field in field[4:featureEnd]]
    # 返回分类特征字段 + 数值特征字段
    return np.concatenate((categoryFeatures, numericalFeatures))

def convert_float(x):
    return (0 if x=="?" else float(x))

def extract_label(field):
    label = field[-1]
    return float(label)

print("生成类别字典")
categorisMap = lines.map(lambda fields: fields[3]).distinct().zipWithIndex().collectAsMap()
print(categorisMap)
labelRDD = lines.map(lambda r: extract_label(r))
featureRDD = lines.map(lambda r: extract_features(r, categorisMap, len(r)-1))
print("数据标准化之前")
for i in featureRDD.first():
    print(i, ",")
print("标准化之后")
stdScaler = StandardScaler(withMean=True, withStd=True).fit(featureRDD)
ScalerFeatureRDD = stdScaler.transform(featureRDD)
for i in ScalerFeatureRDD.first():
    print(i, ",")
print("创建 LabeledPoint 数据，用于给决策树进行训练")
labelpoint=labelRDD.zip(ScalerFeatureRDD)
labelpointRDD = labelpoint.map(lambda r: LabeledPoint(r[0], r[1]))
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
    metrics = BinaryClassificationMetrics(scoreAndLabels)
    return metrics.areaUnderROC

def trainEvaluateModel(trainData, validationData, numIterations, stepSize, miniBatchFraction):
    startTime = time()
    model = LogisticRegressionWithSGD.train(trainData, numIterations, stepSize, miniBatchFraction)
    AUC = evaluateModel(model, validationData)
    duration = time() - startTime
    print("训练评估：numIterations->", numIterations, ", stepSize->", stepSize, ", miniBatchFraction->", miniBatchFraction)
    print("==> 所需时间:", duration, "s ,AUC=", AUC)
    return (AUC, duration, numIterations, stepSize, miniBatchFraction, model)

print("找到最佳参数组合，会计算 4x4x3 =  遍")
print("===========================")
numIterationsList = [3, 5, 10, 15]
stepSizeList = [10, 50, 100]
miniBatchFractionList = [0.5, 0.8, 1.0]

def evalAllParameter(trainData, validationData, numIterationsList, stepSizeList, miniBatchFractionList):
    metrics = [trainEvaluateModel(trainData, validationData, numIterations, stepSize, miniBatchFraction) \
        for numIterations in numIterationsList 
        for stepSize in stepSizeList 
        for miniBatchFraction in miniBatchFractionList ]
    # 找出 AUC 最大的参数组合
    Smetrics = sorted(metrics, key=lambda k:k[0], reverse=True)
    bestParameter = Smetrics[0]
    print("最佳参数：numIterations->", bestParameter[2], ", stepSize->", bestParameter[3], ", miniBatchFraction->", bestParameter[4])
    print("==> 所需时间:", bestParameter[1], "s ,AUC=", bestParameter[0])
    return bestParameter[5]

print("最佳参数组合")
bestModel = evalAllParameter(trainData, validationData, numIterationsList, stepSizeList, miniBatchFractionList)
print("测试结果")
auc = evaluateModel(bestModel, testData)
print("Test AUC=", auc)