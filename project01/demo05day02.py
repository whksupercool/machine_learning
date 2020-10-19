from sklearn.neighbors import KNeighborsClassifier
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
    data['hour'] = time_value.hour
    data['weekday'] = time_value.weekday

    # 三. 特征工程
    return None


if __name__ == '__main__':
    knncls()
