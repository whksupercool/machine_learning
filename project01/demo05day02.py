from sklearn.datasets import load_iris, fetch_20newsgroups, load_boston
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import pandas as pd


def knncls():
    """
    K-近邻预测用户签到位置
    :return: None
    """
    # 一. 读取数据
    data = pd.read_csv("D:/资料/机器学习/数据资料/facebook/train.csv")
    # print(data.head(10))

    # 二. 处理数据
    # 1. 缩小数据,查询数据筛选
    data = data.query("x > 1.0 &  x < 1.25 & y > 2.5 & y < 2.75")

    # 2. 处理时间的数据
    time_value = pd.to_datetime(data["time"], unit="s")
    # print(time_value)

    # 把日期格式转换成字典格式
    time_value = pd.DatetimeIndex(time_value)

    # 3. 构造一些特征
    data["day"] = time_value.day
    data["hour"] = time_value.hour
    data["weekday"] = time_value.weekday
    # 把时间戳特征删除
    data = data.drop(["time"], axis=1)
    # print(data)

    # 把签到数量少于n个目标位置删除
    place_count = data.groupby("place_id").count()
    tf = place_count[place_count.row_id > 3].reset_index()  # reset_index() 重新设置索引，把place_id设置为一列数据
    data = data[data["place_id"].isin(tf.place_id)]
    # print(data)

    # 取出数据当中的目标值（y）和特征值（x）
    y = data["place_id"]
    # x = data.drop(["place_id"], axis=1)
    x = data.drop(["row_id"], axis=1)

    # 进行数据分割的数据集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 三. 特征工程（标准化）
    std = StandardScaler()

    # 对测试集和训练集的特征值进行标准化
    x_train = std.fit_transform(x_train)
    x_test = std.fit_transform(x_test)

    # 进行算法流程
    # knn = KNeighborsClassifier(n_neighbors=5)
    knn = KNeighborsClassifier()

    # # fit, predict, score
    # knn.fit(x_train, y_train)
    #
    # # 得出预测结果
    # y_predict = knn.predict(x_test)
    # print("预测的目标签到位置为：", y_predict)
    #
    # # 得出准确率
    # print("预测的准确率:", knn.score(x_test, y_test))

    # 构造一些参数的值进行搜索
    param = {"n_neighbors": [3, 5, 10]}
    # 进行网格搜索
    gc = GridSearchCV(knn, param_grid=param, cv=10)
    gc.fit(x_train, y_train)

    # 预测准确率
    print("在测试集上准确率：", gc.score(x_test, y_test))
    print("在交叉验证当中最好的结果：", gc.best_score_)
    print("选择最好的模型是：", gc.best_estimator_)
    print("每个超参数每次交叉验证的结果：", gc.cv_results_)

    return None


def naviebayes():
    """
    朴素贝叶斯进行文本分类
    :return: None
    """
    news = fetch_20newsgroups(subset="all")

    # 进行数据分割
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25)

    # 对数据集进行特征抽取
    tf = TfidfVectorizer()
    # 以训练集当中的词的列表进行每篇文章重要性统计
    x_train = tf.fit_transform(x_train)
    # print(tf.get_feature_names())
    x_test = tf.transform(x_test)

    # 进行朴素贝叶斯算法的预测
    mlt = MultinomialNB(alpha=1.0)
    # print(x_train.toarray())
    mlt.fit(x_train, y_train)
    y_predict = mlt.predict(x_test)
    print("预测的文章类别为：", y_predict)

    # 得出准确率
    print("准确率为：", mlt.score(x_test, y_test))

    # 得出精确率和召回率
    print("每个类别的精确率和召回率：", classification_report(y_test, y_predict, target_names=news.target_names))

    return None


if __name__ == '__main__':
    knncls()
    # naviebayes()
