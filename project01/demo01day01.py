from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer  # 自版本0.20以来 SimpleImputer 代替 Imputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import numpy as np
import jieba


def dictvec():
    """
    字典数据抽取
    :return: None
    """
    # 实例化
    dict = DictVectorizer(sparse=False)

    # 调用fit_transform
    data = dict.fit_transform(
        [{'city': '北京', 'temperature': 100}, {'city': '上海', 'temperature': 60}, {'city': '深圳', 'temperature': 30}])

    print(dict.get_feature_names())

    print(dict.inverse_transform(data))

    print(data)

    return None


def countvec():
    """
    对文本进行特征值化
    :return: None
    """
    cv = CountVectorizer()
    data = cv.fit_transform(["人生苦短，我们 喜欢 python", "人生漫长，我们 不用 python"])

    print(cv.get_feature_names())

    print(data.toarray())

    return None


def cut_word():
    """
    分词器
    :return: 
    """

    con1 = jieba.cut("今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。")
    con2 = jieba.cut("我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。")
    con3 = jieba.cut("如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。")

    # 转换成列表
    content1 = list(con1)
    content2 = list(con2)
    content3 = list(con3)

    # 把列表转换成字符串
    c1 = ' '.join(content1)
    c2 = ' '.join(content2)
    c3 = ' '.join(content3)

    return c1, c2, c3


def chinese_characters_vec():
    """
    中文特征值化
    :return: None
    """

    c1, c2, c3 = cut_word()

    cv = CountVectorizer()
    data = cv.fit_transform([c1, c2, c3])

    print(cv.get_feature_names())

    print(data.toarray())

    return None


def tf_idf_vec():
    """
    中文特征值化(频率)
    :return: None
    """

    c1, c2, c3 = cut_word()

    tf = TfidfVectorizer()
    data = tf.fit_transform([c1, c2, c3])

    print(tf.get_feature_names())

    print(data.toarray())

    return None


def mm():
    """
    归一化处理
    :return: None
    """

    mm = MinMaxScaler()
    data = mm.fit_transform([[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]])
    print(data)

    return None


def im():
    """
    缺失值处理
    :return: None
    """

    im = SimpleImputer(missing_values='nan', strategy='mean')

    data = im.fit_transform([[1, 2], [np.nan, 3], [7, 6]])

    print(data)

    return None


def stand():
    """
    标准化缩放
    :return: None
    """
    std = StandardScaler()
    # data = std.fit_transform([[1., -1., 3.], [2., 4., 2.], [4., 6., -1.]])
    data = std.fit_transform([[14488, 7.153469, 1.673904],
                              [26052, 1.441871, 0.805124],
                              [75136, 13.147394, 0.428964],
                              [38344, 1.669788, 0.134296],
                              [72993, 10.141740, 1.032955],
                              [35948, 6.830792, 1.213192],
                              [42666, 13.276369, 0.543880],
                              [67497, 8.631577, 0.749278],
                              [35483, 12.273169, 1.508053],
                              [50242, 3.723498, 0.831917]])
    print(data)

    return None


def var():
    """
    特征选择-删除低方差的特征
    :return: None
    """

    var = VarianceThreshold(threshold=0.0)
    data = var.fit_transform([[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]])
    print(data)

    return None


def pca():
    """
    主成分分析进行特征降维
    :return: 
    """

    pca = PCA(n_components=0.9)
    data = pca.fit_transform([[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]])
    print(data)

    return None


if __name__ == '__main__':
    # dictvec()
    # countvec()
    # chinese_characters_vec()
    # tf_idf_vec()
    # mm()
    # stand()
    # im()
    # var()
    pca()
