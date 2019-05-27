from pyspark.context import SparkContext
from pyspark.conf import SparkConf
from pyspark.mllib.recommendation import Rating, ALS, MatrixFactorizationModel

sc = SparkContext(conf=SparkConf().setAppName("MovieLensRec"))

path = "file:/Users/dawang/Documents/GitHub/computational-advertising-note/spark"
print("数据路径", path)
rawUserData = sc.textFile(path + "/data/ml-100k/u.data")
print("用户数量", rawUserData.count())
print("u.data 第一项数据")
print(rawUserData.first())
print("读取 rawUserData 前 3 个字段，生成 rawRatings")
rawRatings = rawUserData.map(lambda line: line.split("\t")[:3])
print("查看前五个")
print(rawRatings.take(5))

print("准备 ALS 训练数据 Rating RDD，格式为 Rating(user, product, rating)")
ratingsRDD = rawRatings.map(lambda x: (x[0], x[1], x[2]))
print("rating 个数", ratingsRDD.count() ,"。查看前五个")
print(ratingsRDD.take(5))

print("查看不重复的用户数")
numUsers = ratingsRDD.map(lambda x: x[0]).distinct().count()
print(numUsers)

print("查看不重复的电影数")
numMovies = ratingsRDD.map(lambda x: x[1]).distinct().count()
print(numMovies)

print("ALS.train 可以分为显式评分训练与隐式评分训练，我们使用显式")
model = ALS.train(ratingsRDD, 10, 10, 0.01)
print(model)

print("保存一下模型")
try:
    model.save(sc, path + "/model/ALS")
    print("模型已保存")
except Exception:
    print("模型已存在，请先删除")

print("载入一下刚保存的模型")
try:
    model = MatrixFactorizationModel.load(sc, path + "/model/ALS")
    print("已载入模型")
except Exception:
    print("找不到刚保存的模型，请先训练")

print("训练完成。针对用户推荐电影，这里的参数是给用户 100 推荐 5 部电影")
print(model.recommendProducts(100, 5))

print("还可以查看某用户对某用户的评分，这里是计算用户 100 对电影 1141 的评分")
print(model.predict(100, 1141))

print("找到针对某电影感兴趣的用户，可以用于营销，这里的参数是针对电影 200 推荐前 5 个用户")
print(model.recommendUsers(product=200, num=5))

print("前面都是电影的 ID，我们接下来显示出电影的名称")
print("读取 u.item")
itemRDD = sc.textFile(path+"/data/ml-100k/u.item")
print("电影个数", itemRDD.count())
print("创建电影 ID 与名称的字典")
movieTitle = itemRDD.map(lambda line: line.split("|")).map(lambda a:(float(a[0]), a[1])).collectAsMap()
print("字典长度", len(movieTitle))
print("查看前五个")
print(list(movieTitle.items())[:5])
print("显示推荐的电影名称，还是针对用户 100")
recommendP = model.recommendProducts(100, 5)
for p in recommendP:
    print("User", p[0], "->Movie", movieTitle[p[1]], ", rating->", p[2])

print("显示推荐的电影名称，这次针对用户 200")
recommendP = model.recommendProducts(200, 5)
for p in recommendP:
    print("User", p[0], "->Movie", movieTitle[p[1]], ", rating->", p[2])