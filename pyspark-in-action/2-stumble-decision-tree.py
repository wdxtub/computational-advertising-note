from pyspark.context import SparkContext
from pyspark.conf import SparkConf
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
print("创建 LabeledPoint 数据，用于给决策树进行训练")
from pyspark.mllib.regression import LabeledPoint
labelpointRDD = lines.map(lambda r: LabeledPoint(extract_label(r), extract_features(r, categorisMap, len(r)-1)))
print("查看下数据")
print(labelpointRDD.take(1))
print("将数据按照 8：1：1 分配，作为训练、验证和测试")
(trainData, validationData, testData) = labelpointRDD.randomSplit([8,1,1])
print("训练", trainData.count(), ",验证", validationData.count(), ",测试", testData.count())
print("数据暂存到内存中")
trainData.persist()
validationData.persist()
testData.persist()

print("训练模型")
from pyspark.mllib.tree import DecisionTree
model = DecisionTree.trainClassifier(trainData, numClasses=2, \
    categoricalFeaturesInfo={}, impurity="entropy", \
    maxDepth=5, maxBins=5)

print("预测结果")
rawTestDataWithHeader = sc.textFile(path + "/data/stumbleupon/test.tsv")
header = rawTestDataWithHeader.first()
rawTestData = rawTestDataWithHeader.filter(lambda x:x != header)
rTestData = rawTestData.map(lambda x: x.replace("\"", ""))
testLines = rTestData.map(lambda x: x.split("\t"))
print("共", testLines.count(), "项")
dataRDD = testLines.map(lambda r: (r[0], extract_features(r, categorisMap, len(r))))
DescDict = {
    0:"ephemeral",
    1:"evergreen"
}
for data in dataRDD.take(10):
    predictResult = model.predict(data[1])
    print("网址:", data[0], ",预测:",predictResult, ",说明:", DescDict[predictResult])


def evaluateModel(model, validationData):
    score = model.predict(validationData.map(lambda p: p.features))
    scoreAndLabels = score.zip(validationData.map(lambda p: p.label))
    metrics = BinaryClassificationMetrics(scoreAndLabels)
    return metrics.areaUnderROC

def trainEvaluateModel(trainData, validationData, impurityParam, maxDepthParam, maxBinsParam):
    startTime = time()
    model = DecisionTree.trainClassifier(trainData, numClasses=2, categoricalFeaturesInfo={}, \
        impurity=impurityParam, maxDepth=maxDepthParam, maxBins=maxBinsParam)
    AUC = evaluateModel(model, validationData)
    duration = time() - startTime
    print("训练评估：impurity->", impurityParam, ", maxDepth->", maxDepthParam, ", maxBins->", maxBinsParam)
    print("==> 所需时间:", duration, "s ,AUC=", AUC)
    return (AUC, duration, impurityParam, maxDepthParam, maxBinsParam, model)

print("测试评估函数")
(AUC, duration, impurityParam, maxDepthParam, maxBinsParam, model) = \
    trainEvaluateModel(trainData, validationData, "entropy", 5, 5)

def evalParameter(trainData, validationData, evalParam, impurityList, maxDepthList, maxBinsList):
    metrics = [trainEvaluateModel(trainData, validationData, impurity, maxDepth, maxBins) \
        for impurity in impurityList 
        for maxDepth in maxDepthList 
        for maxBins in maxBinsList ]
    if evalParam == "impurity":
        IndexList = impurityList[:]
    elif evalParam == "maxDepth":
        IndexList = maxDepthList[:]
    elif evalParam == "maxBins":
        IndexList = maxBinsList[:]

    df = pd.DataFrame(metrics, index=IndexList, columns=['AUC', 'duration', 'impurity', 'maxDepth', 'maxBins', 'model'])
    print(df)
    

print("评估 impurity 参数")
print("===========================")
impurityList = ["gini", "entropy"]
maxDepthList = [10]
maxBinsList = [10]
evalParameter(trainData, validationData, "impurity", impurityList, maxDepthList, maxBinsList)

print("评估 maxDepth 参数")
print("===========================")
impurityList = ["gini"]
maxDepthList = [3, 5, 10, 15, 20, 25]
maxBinsList = [10]
evalParameter(trainData, validationData, "maxDepth", impurityList, maxDepthList, maxBinsList)

print("评估 maxDepth 参数")
print("===========================")
impurityList = ["gini"]
maxDepthList = [10]
maxBinsList = [3, 5, 10, 50, 100, 200]
evalParameter(trainData, validationData, "maxBins", impurityList, maxDepthList, maxBinsList)

print("找到最佳参数组合，会计算 2x6x6 = 72 遍")
print("===========================")
impurityList = ["gini", "entropy"]
maxDepthList = [3, 5, 10, 15, 20, 25]
maxBinsList = [3, 5, 10, 50, 100, 200]

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